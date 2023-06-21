from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM as FengshenLlama
from fengshen_inner.models.llama.configuration_llama import LlamaConfig as FengshenConfig
from fengshen_inner.models.megatron import mpu
import argparse


def merge_data(module):
    if hasattr(module, "merge"):
        module.merge()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="merge lora weight")
    parser.add_argument(
        "--input_path",
        help="lora model",
    )
    parser.add_argument(
        "--output_path",
        help="Location to write fengshen mode",
    )
    args = parser.parse_args()
    mpu.set_model_parallel_world_size(1)
    mpu.set_model_parallel_rank(0)
    
    model = FengshenLlama.from_pretrained(args.input_path)
    model.apply(merge_data)
    
    model.config.lora = False
    model.save_pretrained(args.output_path)