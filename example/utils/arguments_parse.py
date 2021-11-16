import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_data_path", type=str, help="train file")
parser.add_argument("--dev_data_path", type=str, help="test file")
parser.add_argument("--test_data_path", type=str, help="test file")
parser.add_argument("--pretrained_model_path", type=str, help="pretrained_model_path")
parser.add_argument("--model_type", type=str, default="bert",help="bert")
parser.add_argument("--checkpoints", type=str, help="output_dir")
parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
parser.add_argument("--max_length", type=int, default=512,help="max_length")
parser.add_argument("--epoch", type=int, default=100,help="epoch")
parser.add_argument("--learning_rate", type=float, default=2e-5,help="learning_rate")
parser.add_argument("--clip_norm", type=int, default=0.25,help="clip_norm")
parser.add_argument("--output_path", type=str, help="output")
parser.add_argument("--log_file_path", type=str, default=None,help="task name")

args = parser.parse_args()
