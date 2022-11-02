#coding=utf-8
import os
max_seq_length = 512
data_path = '/cognitive_comp/common_data/floyed/floyed.txt'
out_path = '/cognitive_comp/common_data/floyed/floyed_out.txt'
if os.path.exists(out_path):
    os.remove(out_path)
fout = open(out_path, 'a')
with open(data_path, 'r') as f:
    lines = f.readlines()
    seq_len = 0
    seq = ''
    for line in lines:
        line = line.strip()
        if len(line) > 20:
            if seq_len + len(line) <= max_seq_length:
                seq_len += len(line)
                seq += line
            else:
                if seq_len > 0:
                    fout.write(seq)
                    fout.write('\n')
                seq_len = 0
                seq = ''
fout.close()    


