import argparse
import os
from multiprocessing import Process
from pathlib import Path
import subprocess
import tempfile
import json
import torchaudio
import torch
import torchaudio.functional as F
import soundfile as sf


def transfer(data, target_sample_rate, src_home, tgt_home):
    for audio in data:
        tempfilename = Path(tempfile.gettempdir()).joinpath(next(tempfile._get_candidate_names())+".wav")
        path = audio["path"]
        tgt_dir_name = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])
        tgt_dir_name = os.path.join(tgt_home, tgt_dir_name)
        os.makedirs(tgt_dir_name, exist_ok=True)
        src_path = os.path.join(src_home, path)
        subprocess.run(f"ffmpeg -i {src_path}  -ar 16000 -f wav -y -loglevel 1 {tempfilename}", shell=True)
        wav, sample_rate = torchaudio.backend.sox_io_backend.load(tempfilename)
        wav = torch.squeeze(wav, 0)
        # print(wav.shape)
        texts = dict()
        if sample_rate != target_sample_rate:
            wav = F.resample(wav, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        for segment in audio["segments"]:
            tgt_path = os.path.join(tgt_dir_name, segment["sid"]+".flac")
            if "text" in segment:
                texts[segment["sid"]] = segment["text"]
            stime = segment["begin_time"]
            etime = segment["end_time"]
            ssample = int(stime * target_sample_rate)
            esample = int(etime * target_sample_rate)
            wav_seg = wav[ssample:esample]
            sf.write(tgt_path, wav_seg, target_sample_rate)
        tgt_path = os.path.join(tgt_dir_name, audio["aid"]+".trans.txt")
        with open(tgt_path, "w") as f:
            for seg in texts:
                print(seg+" "+texts[seg], file=f)
        os.remove(tempfilename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--src")
    parser.add_argument("--tgt")
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("-n", type=int, default=64)

    args = parser.parse_args()
    json_path = args.json
    n = args.n
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["audios"]
        print("load data")
        splits = [[] for _ in range(n)]
        for i, item in enumerate(data):
            splits[i % n].append(item)

    process_list = []
    for item in splits:
        process_list.append(
            Process(target=transfer, args=(
                    item, args.target_sample_rate, args.src, args.tgt
                    )
                    )
        )

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()


if __name__ == "__main__":
    main()
