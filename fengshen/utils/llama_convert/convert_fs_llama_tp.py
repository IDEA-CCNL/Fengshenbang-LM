import argparse
import os
import json
import torch
from fengshen.models.llama.configuration_llama import LlamaConfig


__HF_NORM_PREFIX__ = "llama.final_layer_norm"
__HF_EMBED_IN_KEY__ = "llama.embed_in.word_embeddings.weight"
__HF_EMBED_OUT_KEY__ = "embed_out.final_linear.weight"
__HF_LAYER_PREFIX__ = "llama.layers"
__WEIGHT_MAP_FILE__ = "pytorch_model.bin.index.json"


def make_output_dir(path, parallel_size):
    """
    root_dir
    |--- part_0
    |___ part_1
    """
    try:
        os.mkdir(path)
    except:
        pass

    for i in range(parallel_size):
        try:
            os.mkdir(os.path.join(path, f"part_{i}"))
        except:
            pass


def save_splits(input_dir, output_dir, helper, config):
    weight_map_file = os.path.join(input_dir, __WEIGHT_MAP_FILE__)
    with open(weight_map_file, 'r') as fp:
        weight_map = json.load(fp)
    for rank, sd in enumerate(helper.sequential_cache):
        output_part_dir = os.path.join(output_dir, f"part_{rank}")
        with open(os.path.join(output_part_dir, __WEIGHT_MAP_FILE__), 'w') as f:
            json.dump(weight_map, f)
        config.save_pretrained(output_part_dir)
        for file_name, keys in helper.revert_weight_map.items():
            output_sd = {}
            for k in keys:
                if k in sd:
                    output_sd[k] = sd[k]
            torch.save(output_sd, os.path.join(output_part_dir, file_name))


def get_loaders(root_dir, weight_map):
    loaders_map = {}
    weight_map_with_loader = {}
    revert_weight_map = {}
    for k, v in weight_map['weight_map'].items():
        if v in revert_weight_map:
            revert_weight_map[v].append(k)
        else:
            revert_weight_map[v] = [k]
            # 打开对应的state_dict
            ld = torch.load(os.path.join(root_dir, v), map_location='cpu')
            loaders_map[v] = ld
        weight_map_with_loader[k] = loaders_map[v]
    return weight_map_with_loader, revert_weight_map, loaders_map.values()


class Helper:
    def __init__(
            self, args):
        self.num_output_shards = args.model_parallel_size
        self.sequential_cache = [{} for _ in range(args.model_parallel_size)]
        self.init_weight_map(args)

    def init_weight_map(self, args):
        weight_map_file = os.path.join(args.input_dir, __WEIGHT_MAP_FILE__)
        with open(weight_map_file, 'r') as fp:
            weight_map = json.load(fp)
        self.weight_map, self.revert_weight_map, self.loaders = get_loaders(
            args.input_dir, weight_map)

    def del_loaded(self, key: str):
        # Remove from memory as we go along
        if key in self.weight_map:
            del self.weight_map[key][key]

    def shard(self, x, dim):
        x_shape = list(x.shape)
        assert x_shape[dim] % self.num_output_shards == 0
        new_x_shape = (
            x_shape[:dim]
            + [self.num_output_shards, x_shape[dim] // self.num_output_shards]
            + x_shape[dim + 1:]
        )
        x = x.view(*new_x_shape)
        return torch.movedim(x, 0, dim)

    def add_sequential_shard(self, dictionary):
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v[rank].clone()
                self.sequential_cache[rank][k] = v[rank].clone()

    def add_sequential_duplicates(self, dictionary):
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()
                self.sequential_cache[rank][k] = v.clone()

    def add_sequential(self, dictionary, rank):
        for k, v in dictionary.items():
            # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()
            self.sequential_cache[rank][k] = v.clone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="covert hf model to hf model with mp"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to hf model dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to hf model dir",
    )
    parser.add_argument(
        "--model_parallel_size",
        type=int,
        default=1,
        help="Path to hf model dir",
    )
    args = parser.parse_args()
    make_output_dir(args.output_dir, args.model_parallel_size)

    helper = Helper(args)
    config = LlamaConfig.from_pretrained(args.input_dir)

    num_output_shards = args.model_parallel_size
    num_heads_per_output_shard = config.num_attention_heads // num_output_shards
    dims_per_head = config.hidden_size // config.num_attention_heads
    for k, v in helper.weight_map.items():
        # embed in and out
        if k in [__HF_EMBED_IN_KEY__, __HF_EMBED_OUT_KEY__]:
            helper.add_sequential_shard({k: helper.shard(v[k], dim=0)})
        elif k.startswith(__HF_NORM_PREFIX__):
            helper.add_sequential_duplicates({k: v[k]})

        elif k.startswith(__HF_LAYER_PREFIX__):
            # QKV weight and bias
            if k.find("query_key_value") != -1:
                output_shape = [num_output_shards, num_heads_per_output_shard *
                                3 * dims_per_head] + list(v[k].shape[1:])
                sharded = v[k].view(output_shape)
                for out_rank in range(num_output_shards):
                    helper.add_sequential({k: sharded[out_rank]}, out_rank)
            # rotary emb
            elif k.find("rotary_emb.inv_freq") != -1:
                helper.add_sequential_duplicates({k: v[k]})
            # layer_norm
            elif k.find("layernorm") != -1:
                helper.add_sequential_duplicates({k: v[k]})
            # linear
            elif k.find("dense") != -1 or k.find("mlp") != -1:
                # 纵切
                if k.find("w2") != -1 or k.find("attention") != -1:
                    if k.find('weight') != -1:
                        shard = helper.shard(v[k], dim=1)
                        helper.add_sequential_shard({k: shard})
                    # bias不切
                    else:
                        helper.add_sequential_duplicates({k: v[k]})
                # 横切
                else:
                    shard = helper.shard(v[k], dim=0)
                    helper.add_sequential_shard({k: shard})
            else:
                print(f"WARNING: unexcept key {k}")
        else:
            print(f"WARNING: unexcept key {k}")

        helper.del_loaded(k)

    save_splits(args.input_dir, args.output_dir, helper, config)
