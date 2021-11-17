# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_data_path", type=str, help="train file")
parser.add_argument("--dev_data_path", type=str, help="test file")
parser.add_argument("--test_data_path", type=str, help="test file")
parser.add_argument("--pretrained_model_path", type=str,
                    help="pretrained_model_path")
parser.add_argument("--model_type", type=str, default="megatron", help="megatron or roformer")
parser.add_argument("--checkpoints", type=str, help="checkpoint")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--max_length", type=int, default=512, help="max_length")
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--learning_rate", type=float,
                    default=2e-5, help="learning_rate")
parser.add_argument("--clip_norm", type=int, default=0.25, help="clip_norm")
parser.add_argument("--output_path", type=str, help="output")
parser.add_argument("--log_file_path", type=str,
                    default=None, help="log file save path")

args = parser.parse_args()
