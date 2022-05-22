import sys
import os
from tqdm import tqdm
sys.path.append('../../')

if __name__ == '__main__':
    from data.fs_datasets import load_dataset
    dataset = load_dataset('wudao_180g', num_proc=100)
    print('dataset loaded', flush=True)

    shuffle_ds = dataset['train'].shuffle(seed=42, writer_batch_size=1000)
    print('dataset shuffled', flush=True)
    need_size = len(shuffle_ds)

    f = open('shuffle_corpus_{}.txt'.format(need_size), 'w', encoding='utf-8')
    for i in tqdm(range(0, need_size)):
        f.write(shuffle_ds[i]['text'] + os.linesep)
    f.close()
