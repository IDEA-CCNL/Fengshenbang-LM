import json

import datasets


_CITATION = """\
@article{talmor2021multimodalqa,
  title={MultiModalQA: Complex Question Answering over Text, Tables and Images},
  author={Talmor, Alon and Yoran, Ori and Catav, Amnon and Lahav, Dan and Wang, Yizhong and Asai, Akari and Ilharco, Gabriel and Hajishirzi, Hannaneh and Berant, Jonathan},
  journal={arXiv preprint arXiv:2104.06039},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the MMQA.
"""

_HOMEPAGE = "https://github.com/allenai/multimodalqa"

_LICENSE = "MIT License"

_URL = "https://github.com/allenai/multimodalqa/raw/master/dataset/"
_TRAINING_FILE = "MMQA_train.jsonl.gz"
_DEV_FILE = "MMQA_dev.jsonl.gz"
_TEST_FILE = "MMQA_test.jsonl.gz"
_TEXTS_FILE = "MMQA_texts.jsonl.gz"
_TABLES_FILE = "MMQA_tables.jsonl.gz"
_PASSAGE_FILE = "MMQA_texts.jsonl.gz"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    # "test": f"{_URL}{_TEST_FILE}",
    "texts": f"{_URL}{_TEXTS_FILE}",
    "tables": f"{_URL}{_TABLES_FILE}",
    "passages": f"{_URL}{_PASSAGE_FILE}"
}

class MMQA(datasets.GeneratorBasedBuilder):
    """The MMQA dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "table_id": datasets.features.Sequence(datasets.Value("string")),
                    "table": datasets.features.Sequence(
                        {"header": datasets.features.Sequence(datasets.Value("string")),
                         "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))}
                    ),
                    "context": datasets.features.Sequence(datasets.Value("string")),
                    "meta": datasets.features.Sequence(datasets.Value("string")),
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
        tables = {}
        with open(tablepath, 'r') as f:
            for line in f:
                table = json.loads(line)
                tables[table["id"]] = table
        texts = {}
        with open(passagepath, 'r') as f:
            for line in f:
                text = json.loads(line)
                texts[text["id"]] = text
        with open(filepath, 'r') as f:
            count = 0
            for idx, line in enumerate(f):
                example = json.loads(line)
                if example["metadata"]["type"] in ["TableQ", "TextQ", "Compose(TextQ,TableQ)",
                                                   "Intersect(TableQ,TextQ)", "Compare(TableQ,Compose(TableQ,TextQ))",
                                                   "Compose(TableQ,TextQ)"]:
                    count += 1
                    supporting_tables = []
                    supporting_texts = []
                    for context in example["supporting_context"]:
                        if context["doc_part"] == "text":
                            doc_id = context["doc_id"]
                            text = texts[doc_id]["title"] + ". " + texts[doc_id]["text"]
                            supporting_texts.append(text)
                        elif context["doc_part"] == "table":
                            doc_id = context["doc_id"]
                            table_page_title = tables[doc_id]["title"]
                            table_caption = tables[doc_id]["table"]["table_name"]
                            table_header = [column["column_name"] for column in tables[doc_id]["table"]["header"]]
                            table_rows = [[cell["text"] for cell in row] for row in
                                          tables[doc_id]["table"]["table_rows"]]
                            supporting_tables.append({
                                "doc_id": doc_id,
                                "title": table_page_title,
                                "caption": table_caption,
                                "header": table_header,
                                "rows": table_rows
                            })
                        else:
                            raise ValueError("Unknown doc_part type: {}".format(context["doc_part"]))
                    yield count, {
                        "id": example["qid"],
                        "question": example["question"],
                        "table_id": [table["doc_id"] for table in supporting_tables],
                        "table": [{"header": table["header"], "rows": table["rows"]} for table in supporting_tables],
                        "context": supporting_texts,
                        "meta": [table["title"] + " | " + table["caption"] for table in supporting_tables],
                        "answer_text": " | ".join([str(answer["answer"]) for answer in example["answers"]]),
                    }