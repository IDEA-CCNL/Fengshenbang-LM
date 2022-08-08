import numpy as np
import math
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset


import numpy as np
import math
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset


class PropMixingRandomSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets) # 所有读取的datasets长度总和
        # self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.cumulative_dataset_size = self.dataset.cumsum([cur_dataset for cur_dataset in dataset.datasets])
        self.upper_limit = 2**15 # K
        

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def _mixing_rate(self,samplers_list):
        # example-proportional mixing 
        # m-th dataset in n dataset, prob rm = min(em, K)/\sum min(en,K), K is hyperpara
        # https://www.jmlr.org/papers/volume21/20-074/20-074.pdf
        dataset_len = [sampler.__len__() for sampler in samplers_list]
        dataset_len_new = [min(dataset_len[idx], self.upper_limit) for idx in range(self.number_of_datasets)]
        dataset_prob = [each / sum(dataset_len_new) for each in dataset_len_new]#i.e. rm
        return dataset_prob

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        # shuffle each dataset and get sampler_iterators
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset) 
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        # first data index in large dataset
        # dataset_offset = [0] + self.dataset.cumulative_sizes[:-1]
        # fix bug: concat dataset cum_size = [3,6] because don't assign field name
        dataset_offset = [0] + self.cumulative_dataset_size
        dataset_prob = self._mixing_rate(samplers_list)

        step = self.upper_limit
        samples_to_grab = int(step / self.batch_size)
        epoch_samples = self.upper_limit * self.number_of_datasets
     
        # indexes from the combined dataset
        final_samples_list = []  
        for _ in range(0, epoch_samples, step):
            # sample by prob
            sampler_idx = np.random.choice(self.number_of_datasets,p=dataset_prob)
            print(sampler_idx)
            cur_batch_sampler = sampler_iterators[sampler_idx]
            cur_samples = []
            i = sampler_idx 

            for _ in range(samples_to_grab):
                try: # if exceed, __next__ will return StopIteration
                    cur_sample_org = cur_batch_sampler.__next__()
                    print(cur_sample_org)
                    cur_sample = cur_sample_org + dataset_offset[i]
                    cur_samples.append(cur_sample)
                except StopIteration: # extend small dataset
                    print("excecption")
                    sampler_iterators[i] = samplers_list[i].__iter__()
                    cur_batch_sampler = sampler_iterators[i]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + dataset_offset[i]
                    cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

class TempMixingRandomSampler(PropMixingRandomSampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, temperture=1.0):
        super().__init__(dataset, batch_size)
        self.temperature = temperture

    def _mixing_rate(self, samplers_list):
        dataset_len = [sampler.__len__() for sampler in samplers_list]
        dataset_len_new = [min(dataset_len[idx], self.upper_limit) for idx in range(self.number_of_datasets)]
        dataset_prob = [each / sum(dataset_len_new) for each in dataset_len_new]#i.e. rm
        # temperature
        for rm in dataset_prob:
            dataset_prob_temp = rm ** (1.0 / self.temperature)
        
        return dataset_prob_temp
