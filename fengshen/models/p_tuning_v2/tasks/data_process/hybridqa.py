import json

import datasets
import os
import nltk

_CITATION = """\
@article{chen2020hybridqa,
  title={Hybridqa: A dataset of multi-hop question answering over tabular and textual data},
  author={Chen, Wenhu and Zha, Hanwen and Chen, Zhiyu and Xiong, Wenhan and Wang, Hong and Wang, William},
  journal={arXiv preprint arXiv:2004.07347},
  year={2020}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the HybridQA.
"""

_HOMEPAGE = "https://github.com/wenhuchen/HybridQA"

_LICENSE = "MIT License"

_URL = "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/"
_TRAINING_FILE = "train.traced.json"
_DEV_FILE = "dev.traced.json"
_CONTEXT_FILE_URL = "https://github.com/wenhuchen/WikiTables-WithLinks/archive/refs/heads/master.zip"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "context": _CONTEXT_FILE_URL,
}

WINDOW_SIZE = 3

class HybridQA(datasets.GeneratorBasedBuilder):
    """The Hybrid dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                              "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
                    "passage": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answer_text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"], "contextpath": downloaded_files["context"]}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"], "contextpath": downloaded_files["context"]}),
        ]

    def _generate_examples(self, filepath, contextpath):
        """Yields examples."""
        # data_id, question, table_id, gold_result_str
        table_tok_path = os.path.join(contextpath, "WikiTables-WithLinks-master", "tables_tok")
        passage_tok_path = os.path.join(contextpath, "WikiTables-WithLinks-master", "request_tok")
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, example in enumerate(data):
                answer_node = example["answer-node"]
                table_id = example["table_id"]
                table = json.load(open(os.path.join(table_tok_path, "{}.json".format(table_id))))
                passages = json.load(open(os.path.join(passage_tok_path, "{}.json".format(table_id))))
                answer = example["answer-text"]
                # how to construct context?
                # keep all cells and appending the sentences that contains answer span into the cell
                header, data, passage_context_str = self.construct_expanded_table(table, passages, answer, answer_node)
                yield idx, {
                    "id": example["question_id"],
                    "question": example["question"],
                    "table_id": example["table_id"],
                    "table": {"header": header, "rows": data},
                    "passage": passage_context_str,
                    "context": table["title"] + " | " + table["section_title"] + " | " + table["section_text"] + " | " + table["intro"],
                    "answer_text": example["answer-text"],
                }

    def construct_expanded_table(self, table, passages, answer, answer_nodes):
        def process_link(link):
            return link.split("/")[-1].replace("_", " ")
        header = [column[0] for column in table["header"]]
        data = [[cell[0] for cell in row] for row in table["data"]]
        selected_passage = {}
        for answer_node in answer_nodes:
            link = answer_node[2]
            type_ = answer_node[3]
            if type_ == "passage":
                # Get passage and locate the sentence of answer
                passage_text = passages[link]
                sents = nltk.sent_tokenize(passage_text)
                has_answer_sent_idx = -1
                for idx, sent in enumerate(sents):
                    if " " + answer.lower() + " " in " " + sent.lower() + " ":
                        has_answer_sent_idx = idx
                selected_sents = sents[max(0, has_answer_sent_idx-(WINDOW_SIZE-1)//2): min(len(sents)-1, has_answer_sent_idx+(WINDOW_SIZE-1)//2)]
                selected_passage[process_link(link)] =  " ".join(selected_sents)
            else:
                pass
        # linearize selected passgae
        passage_context_str = "passages: "
        for key in selected_passage:
            passage_context_str += "{}: {} | ".format(key, selected_passage[key])
        return header, data, passage_context_str

