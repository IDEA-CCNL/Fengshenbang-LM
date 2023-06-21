from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from fengshen.models.megatron import mpu
from fengshen.models.llama.modeling_llama import LlamaForCausalLM as FengshenLlama
from fengshen.models.llama.configuration_llama import LlamaConfig as FengshenConfig
import argparse
import torch
from tqdm import tqdm


def convert_config(fs_config: FengshenConfig):
    
    hf_config = LlamaConfig(
        vocab_size=fs_config.vocab_size,
        hidden_size=fs_config.hidden_size,
        intermediate_size=fs_config.intermediate_size,
        num_hidden_layers=fs_config.num_hidden_layers,
        num_attention_heads=fs_config.num_attention_heads,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=fs_config.rms_norm_epsilon,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    return hf_config


def merge_data(module):
    if hasattr(module, "merge"):
        module.merge()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fengshen llama to hugginface format.")
    parser.add_argument(
        "--input_path",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_path",
        help="Location to write fengshen mode",
    )
    args = parser.parse_args()
    mpu.set_model_parallel_world_size(1)
    mpu.set_model_parallel_rank(0)
    fs_model = FengshenLlama.from_pretrained(args.input_path)
    fs_model.apply(merge_data)
    tokenizer = LlamaTokenizer.from_pretrained(args.input_path)
    fs_config = fs_model.config
    hf_config = convert_config(fs_config)
    hf_model = LlamaForCausalLM(hf_config)

    # embed_in
    hf_model.model.embed_tokens.load_state_dict(
        {"weight": fs_model.llama.embed_in.word_embeddings.weight}
    )

    # embed_out
    hf_model.lm_head.load_state_dict({"weight": fs_model.embed_out.final_linear.weight})

    # final_norm
    hf_model.model.norm.load_state_dict({"weight": fs_model.llama.final_layer_norm.scale})

    num_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    dims_per_head = hidden_size // num_heads

    # layer
    for layer_i in tqdm(range(fs_config.num_hidden_layers)):
        hf_layer = hf_model.model.layers[layer_i]
        fs_layer = fs_model.llama.layers[layer_i]
        
        state_dict = {}
        sharded_qkv = fs_layer.attention.query_key_value.weight.view(num_heads, 3, dims_per_head, hidden_size)
        q, k, v = sharded_qkv.chunk(3, dim=1)
        state_dict["self_attn.q_proj.weight"] = q.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.k_proj.weight"] = k.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.v_proj.weight"] = v.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.o_proj.weight"] = fs_layer.attention.dense.weight
        
        # Just take one
        state_dict["self_attn.rotary_emb.inv_freq"] = fs_layer.attention.rotary_emb.inv_freq

        ## average layernorm stats over mp ranks
        state_dict["input_layernorm.weight"] = fs_layer.input_layernorm.scale
        state_dict["post_attention_layernorm.weight"] = fs_layer.post_attention_layernorm.scale

        ## mlp params
        state_dict["mlp.gate_proj.weight"] = fs_layer.mlp.w1.weight
        state_dict["mlp.up_proj.weight"] = fs_layer.mlp.w3.weight
        state_dict["mlp.down_proj.weight"] = fs_layer.mlp.w2.weight

        ## load state_dict into layer
        hf_layer.load_state_dict(state_dict)

    hf_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

