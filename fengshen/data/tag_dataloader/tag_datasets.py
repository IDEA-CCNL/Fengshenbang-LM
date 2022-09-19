from torch.utils.data import Dataset
from fengshen.metric.utils_ner import get_entities

import os

def get_labels(decode_type):
    with open("/cognitive_comp/lujunyu/data_zh/NER_Aligned/weibo/labels.txt") as f:
        label_list = ["[PAD]", "[START]", "[END]"]

        if decode_type=="crf" or decode_type=="linear":
            for line in f.readlines():
                label_list.append(line.strip())
        elif decode_type=="biaffine" or decode_type=="span":
            for line in f.readlines():
                tag = line.strip().split("-")
                if len(tag) == 1 and tag[0] not in label_list:
                    label_list.append(tag[0])
                elif tag[1] not in label_list:
                    label_list.append(tag[1])
    
    label2id={label:id for id,label in enumerate(label_list)}
    id2label={id:label for id,label in enumerate(label_list)}
    return label2id, id2label

class DataProcessor(object):
    def __init__(self, data_dir, decode_type) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.decode_type = decode_type

    def get_examples(self, mode):
        return self._create_examples(self._read_text(os.path.join(self.data_dir, mode + ".all.bmes")), mode)

    def get_labels(self):
        with open(os.path.join(self.data_dir, "labels.txt")) as f:
            label_list = ["[PAD]", "[START]", "[END]"]

            if self.decode_type=="crf" or self.decode_type=="linear":
                for line in f.readlines():
                    label_list.append(line.strip())
            elif self.decode_type=="biaffine" or self.decode_type=="span":
                for line in f.readlines():
                    tag = line.strip().split("-")
                    if len(tag) == 1 and tag[0] not in label_list:
                        label_list.append(tag[0])
                    elif tag[1] not in label_list:
                        label_list.append(tag[1])

        label2id = {label: i for i, label in enumerate(label_list)}
        return label2id

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entities(labels, id2label=None, markup='bioes')
            examples.append({'guid':guid, 'text_a':text_a, 'labels':labels, 'subject':subject})
        return examples

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines


class TaskDataset(Dataset):
    def __init__(self, processor, mode='train'):
        super().__init__()
        self.data = self.load_data(processor, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, processor, mode):
        examples = processor.get_examples(mode)
        return examples