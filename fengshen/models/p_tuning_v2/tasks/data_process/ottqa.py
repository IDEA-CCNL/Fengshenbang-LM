import json
import nltk
import datasets

_CITATION = """\
@article{chen2020open,
  title={Open question answering over tables and text},
  author={Chen, Wenhu and Chang, Ming-Wei and Schlinger, Eva and Wang, William and Cohen, William W},
  journal={arXiv preprint arXiv:2010.10439},
  year={2020}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the OTT-QA.
"""

_HOMEPAGE = "https://ott-qa.github.io"

_LICENSE = "MIT License"

_URL = "https://github.com/wenhuchen/OTT-QA/raw/a14ec408b2c22e24a44622b01e4242d95b7ecf08/released_data/"
_TRAINING_FILE = "train.traced.json"
_DEV_FILE = "dev.traced.json"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "tables": "https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json",
    "passages": "https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json",
}

WINDOW_SIZE = 3

class OTTQA(datasets.GeneratorBasedBuilder):
    """The OTTQA dataset"""

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
                gen_kwargs={"filepath": downloaded_files["train"], "tablepath": downloaded_files["tables"], "passagepath": downloaded_files["passages"]}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"], "tablepath": downloaded_files["tables"], "passagepath": downloaded_files["passages"]}),
           ]

    def _generate_examples(self, filepath, tablepath, passagepath):
        """Yields examples."""
        # data_id, question, table_id, gold_result_str

        with open(tablepath, encoding="utf-8") as f:
            tables = json.load(f)
        with open(passagepath, encoding="utf-8") as f:
            passages = json.load(f)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, example in enumerate(data):
                table = tables[example["table_id"]]
                answer_node = example["answer-node"]
                answer = example["answer-text"]
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
                selected_sents = sents[max(0, has_answer_sent_idx - (WINDOW_SIZE - 1) // 2): min(len(sents) - 1,
                                                                                                 has_answer_sent_idx + (
                                                                                                             WINDOW_SIZE - 1) // 2)]
                selected_passage[process_link(link)] = " ".join(selected_sents)
            else:
                pass
        # linearize selected passgae
        passage_context_str = "passages: "
        for key in selected_passage:
            passage_context_str += "{}: {} | ".format(key, selected_passage[key])
        return table["header"], table["data"], passage_context_str