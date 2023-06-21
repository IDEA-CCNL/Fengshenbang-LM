from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from fengshen.models.megatron import mpu
from fengshen.models.llama.modeling_llama import LlamaForCausalLM as FengshenLlama
from fengshen.models.llama.configuration_llama import LlamaConfig as FengshenConfig
import argparse
import torch
from tqdm import tqdm


def convert_config(hf_config: LlamaConfig):
    fs_config = FengshenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rotary_pct=1,
        rotary_emb_base=10000,
        max_position_embeddings=hf_config.max_position_embeddings,
        initializer_range=hf_config.initializer_range,
        rms_norm_epsilon=hf_config.rms_norm_eps,
        torch_dtype=hf_config.torch_dtype,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=False,
    )
    fs_config.llama_mlp_multiple_of = 256
    assert fs_config.intermediate_size % fs_config.llama_mlp_multiple_of == 0, \
        f"{fs_config.intermediate_size} % {fs_config.llama_mlp_multiple_of}"
    fs_config.init_method = "small_init"
    fs_config.hidden_dropout = 0
    fs_config.output_layer_init_method = "wang_init"
    fs_config.pos_emb = "rotary"
    fs_config.norm = "rmsnorm"
    fs_config.gpt_j_residual = False
    fs_config.gpt_j_tied = False
    fs_config.apply_query_key_layer_scaling = False
    fs_config.attention_softmax_in_fp32 = False
    fs_config.scaled_masked_softmax_fusion = True
    fs_config.scaled_upper_triang_masked_softmax_fusion = False
    fs_config.bias_gelu_fusion = False
    fs_config.attention_dropout = 0
    fs_config.output_layer_parallelism = "column"
    fs_config.eod_mask_loss = False
    fs_config.bias_dropout_fusion = False
    fs_config.attention_config = [[["flash"], "all"]]
    fs_config.mlp_type = "llama"
    fs_config.use_bias_in_attn_linear = False
    fs_config.lora = False
    return fs_config

def find_closest_multiple(current_num, n):
    if current_num % n == 0:
        return current_num
    closest_multiple = ((current_num // n) + 1) * n
    return closest_multiple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw LLaMA checkpoints to fengshen format.")
    parser.add_argument(
        "--input_path",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_path",
        help="Location to write fengshen mode",
    )
    parser.add_argument(
        "--multiplier",
        default = 1,
        help="Make embedding_size an integer multiple of multiplier",
    )
    args = parser.parse_args()
    hf_model = LlamaForCausalLM.from_pretrained(args.input_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.input_path, use_fast=False)
    hf_config = hf_model.config
    fs_config = convert_config(hf_config)
    # used for FengshenLlama initialized
    mpu.set_model_parallel_world_size(1)
    mpu.set_model_parallel_rank(0)
    mpu.set_init_params_in_cuda(False)
    fs_model = FengshenLlama(fs_config)

    # embed_in
    fs_model.llama.embed_in.load_state_dict(
        {"word_embeddings.weight": hf_model.model.embed_tokens.weight}
    )

    # embed_out
    fs_model.embed_out.load_state_dict(
        {"final_linear.weight": hf_model.lm_head.weight}
    )

    fs_model.resize_token_embeddings(find_closest_multiple(fs_model.config.vocab_size, int(args.multiplier)))

    # final_norm
    fs_model.llama.final_layer_norm.load_state_dict(
        {"scale": hf_model.model.norm.weight}
    )

    num_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    dims_per_head = hidden_size // num_heads

    def permute_rotary(w):
        assert w.shape == (num_heads, dims_per_head, hidden_size)
        return w.view(num_heads, dims_per_head // 2, 2, hidden_size) \
            .transpose(1, 2) \
            .reshape(num_heads, dims_per_head, hidden_size)
    # layer
    for layer_i in tqdm(range(fs_config.num_hidden_layers)):
        fs_layer = fs_model.llama.layers[layer_i]
        hf_layer = hf_model.model.layers[layer_i]
        # Linear
        attn_wo = hf_layer.self_attn.o_proj.weight
        mlp_w1 = hf_layer.mlp.gate_proj.weight
        mlp_w2 = hf_layer.mlp.down_proj.weight
        mlp_w3 = hf_layer.mlp.up_proj.weight

        # Attention
        w_q = hf_layer.self_attn.q_proj.weight.view(num_heads, dims_per_head, hidden_size)
        w_k = hf_layer.self_attn.k_proj.weight.view(num_heads, dims_per_head, hidden_size)
        w_v = hf_layer.self_attn.v_proj.weight.view(num_heads, dims_per_head, hidden_size)
        sharded_qkv = torch.stack([w_q, w_k, w_v], dim=1)
        sharded_qkv = sharded_qkv.view(num_heads*dims_per_head*3, hidden_size)
        # Duplicated
        input_layernorm = hf_layer.input_layernorm.weight
        post_attention_layernorm = hf_layer.post_attention_layernorm.weight
        rotary_inv = hf_layer.self_attn.rotary_emb.inv_freq

        fs_layer.load_state_dict({
            "attention.query_key_value.weight": sharded_qkv,
            # Sharded layers
            "attention.dense.weight": attn_wo.clone(),
            "mlp.w1.weight": mlp_w1.clone(),
            "mlp.w2.weight": mlp_w2.clone(),
            "mlp.w3.weight": mlp_w3.clone(),
            # Duplicated layers
            "input_layernorm.scale": input_layernorm,
            "post_attention_layernorm.scale": post_attention_layernorm,
            "attention.rotary_emb.inv_freq": rotary_inv,
        })

    fs_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
