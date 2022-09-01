# %%

import torch
import sys
sys.path.append("/home/yangqi/Fengshenbang-LM/fengshen/data/")
from fs_datasets import load_dataset
from sampler import PropMixingRandomSampler, TempMixingRandomSampler
from torch.utils.data.dataset import ConcatDataset

batch_size = 8

dataset1 = load_dataset("dial",subname="duconv")
dataset2 = load_dataset("dial",subname="kdconv")
concat_dataset = {
    "train":ConcatDataset([dataset1["train"],dataset2["train"]])
}
# dataloader with BatchSchedulerSampler


# %%
dataset1 = load_dataset("unidial",subname="naturalconv")
dataset2 = load_dataset("unidial",subname="lccc")

# %%
len(concat_dataset["train"])

# %%
sampler=PropMixingRandomSampler(dataset=concat_dataset,batch_size=batch_size)
sampler2 = TempMixingRandomSampler(dataset=concat_dataset,batch_size=batch_size,temperture=0.9)

# %%
import torch
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         sampler=sampler,
                                         batch_size=batch_size,
                                         shuffle=False)
for data in dataloader:
    print(data)
    break

# %%
#len(concat_dataset.datasets[0]["train"])
dl = [d["train"] for d in concat_dataset.datasets]
concat_dataset.cumsum(dl)

# %%
sampler.cumulative_dataset_size

# %%



