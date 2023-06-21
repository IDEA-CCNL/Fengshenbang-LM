import argparse
import os
import json
import torch
from fengshen_inner.models.llama.configuration_llama import LlamaConfig as FengshenConfig
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM as FengshenLlama
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from fengshen_inner.models.megatron import mpu
from glob import glob
import copy
from tqdm import tqdm

__FS_FINAL_NORM_KEY__ = "llama.final_layer_norm.scale"
__FS_EMBED_IN_KEY__ = "llama.embed_in.word_embeddings.weight"
__FS_EMBED_OUT_KEY__ = "embed_out.final_linear.weight"
__FS_LAYER_PREFIX__ = "llama.layers"


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
        torch_dtype=fs_config.torch_dtype,
    )
    return hf_config


def merge_data(module):
    if hasattr(module, "merge"):
        module.merge()


def get_loaders(root_path, mp_size, fs_config):
    fs_model = FengshenLlama(fs_config)
    loaders = []
    for mp in range(mp_size):
        file = os.path.join(root_path, f"mp_rank_{mp:02}_model_states.pt")
        print(f"loading {file}")
        sd = torch.load(file, map_location='cpu')
        new_sd = {}
        for k, v in sd["module"].items():
            try:
                anchor = k.index('llama')
            except:
                if 'embed_out' in k:
                    anchor = k.index('embed_out')
                else:
                    anchor = 0
            rep = k[:anchor]
            new_sd[k.replace(rep, "")] = v
            # new_sd[k.replace("module.model.", "")] = v
        fs_model.load_state_dict(new_sd)
        fs_model.apply(merge_data)
        loaders.append(copy.deepcopy(fs_model.state_dict()))
    return loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="covert hf model to gxy hf model with mp"
    )
    # fs结构的预训练配置
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to hf pretrained model dir",
    )
    # 模型并行数
    parser.add_argument(
        "--model_parallel_size",
        type=int,
        default=1,
        help="Path to hf model dir",
    )
    # lightning checkpoint目录--pretrained_model_path
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to lightning checkpoint dir",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to hf model dir",
    )
    args = parser.parse_args()
    mpu.set_model_parallel_world_size(args.model_parallel_size)
    #mpu.set_init_params_in_cuda(False)
    mpu.set_model_parallel_rank(0)

    fs_config = FengshenConfig.from_pretrained(args.pretrained_model_path)

    loaded_tp_ranks = get_loaders(args.ckpt_path, args.model_parallel_size, fs_config)

    config = convert_config(fs_config)
    tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_model_path)
    num_output_shards = 1
    num_heads_per_output_shard = config.num_attention_heads
    dims_per_head = config.hidden_size // config.num_attention_heads

    hf_model = LlamaForCausalLM(config)

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    dims_per_head = hidden_size // num_heads
    mp_partitions = args.model_parallel_size

    # EMBED_IN
    hf_model.model.embed_tokens.load_state_dict(
        {"weight": torch.cat([t[__FS_EMBED_IN_KEY__] for t in loaded_tp_ranks], dim=0)})
    # EMBED_OUT
    hf_model.lm_head.load_state_dict(
        {"weight": torch.cat([t[__FS_EMBED_OUT_KEY__] for t in loaded_tp_ranks], dim=0)})
    # FINAL_LAYER_NORM
    hf_model.model.norm.load_state_dict(
        {"weight": (sum([t[__FS_FINAL_NORM_KEY__] for t in loaded_tp_ranks])) / mp_partitions})
    # layer
    for layer_i in tqdm(range(config.num_hidden_layers)):
        hf_layer = hf_model.model.layers[layer_i]
        state_dict = {}

        sharded_qkv = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.query_key_value.weight"] for t in loaded_tp_ranks], dim=0)
        sharded_qkv = sharded_qkv.view(num_heads, 3, dims_per_head, hidden_size)
        q, k, v = sharded_qkv.chunk(3, dim=1)
        state_dict["self_attn.q_proj.weight"] = q.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.k_proj.weight"] = k.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.v_proj.weight"] = v.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.o_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.dense.weight"] for t in loaded_tp_ranks], dim=1)
        state_dict["self_attn.rotary_emb.inv_freq"] = \
            loaded_tp_ranks[0][f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.rotary_emb.inv_freq"]

        # average layernorm stats over mp ranks
        state_dict["input_layernorm.weight"] = (sum(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.input_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions
        state_dict["post_attention_layernorm.weight"] = (sum(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.post_attention_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions

        # mlp params
        state_dict["mlp.gate_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w1.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.up_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w3.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.down_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w2.weight"] for t in loaded_tp_ranks], dim=1)

        # load state_dict into layer
        hf_layer.load_state_dict(state_dict)

    hf_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
