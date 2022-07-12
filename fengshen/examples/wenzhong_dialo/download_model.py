from huggingface_hub import snapshot_download
import argparse

total_parser = argparse.ArgumentParser()
total_parser.add_argument('-n', default="bert-base-cased", type=str)
total_parser.add_argument('-p', default="/cognitive_comp/yangqi/model", type=str)
args = total_parser.parse_args()
snapshot_download(repo_id=args.n, cache_dir=args.p, ignore_regex=["*.h5", "*.ot", "*.msgpack"])