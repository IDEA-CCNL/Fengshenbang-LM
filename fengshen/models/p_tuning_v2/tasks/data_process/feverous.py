import json
import os
import sqlite3
from collections import defaultdict

import datasets
from wikiextractor.extract import Extractor, ignoreTag, resetIgnoredTags

_CITATION = """\
@article{aly2021feverous,
  title={FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information},
  author={Aly, Rami and Guo, Zhijiang and Schlichtkrull, Michael and Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Cocarascu, Oana and Mittal, Arpit},
  journal={arXiv preprint arXiv:2106.05707},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the FEVEROUS.
"""

_HOMEPAGE = "https://fever.ai/dataset/feverous.html"

_LICENSE = ""

_URL = "https://fever.ai/download/feverous/"
_TRAINING_FILE = "feverous_train_challenges.jsonl"
_DEV_FILE = "feverous_dev_challenges.jsonl"
_DATABASE = "feverous-wiki-pages-db.zip"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "database": f"{_URL}{_DATABASE}",
}

EVIDENCE_TYPES = ["sentence", "cell", "header_cell", "table_caption", "item"]

extractor = Extractor(0, '', [], '', '')


def clean_markup(markup, keep_links=False, ignore_headers=True):
    """
    Clean Wikimarkup to produce plaintext.

    :param keep_links: Set to True to keep internal and external links
    :param ignore_headers: if set to True, the output list will not contain
    headers, only

    Returns a list of paragraphs (unicode strings).
    """

    if not keep_links:
        ignoreTag('a')

    # returns a list of strings
    paragraphs = extractor.clean_text(markup)
    resetIgnoredTags()

    if ignore_headers:
        paragraphs = filter(lambda s: not s.startswith('## '), paragraphs)

    return " ".join(list(paragraphs))


def get_table_id(meta):
    """
    meta types:
    - table_caption_18
    - cell_18_1_1
    - header_cell_18_0_0
    """
    if meta.startswith("table_caption"):
        return meta.split("_")[-1]
    if meta.startswith("header_cell") or meta.startswith("cell"):
        return meta.split("_")[-3]


def get_list_id(meta):
    """"
    meta types:
    - item_4_25
    """
    return meta.split("_")[1]


def process_table(table):
    rows = []
    for row in table["table"]:
        cells = []
        for cell in row:
            cells.append(clean_markup(cell["value"]))
        rows.append(cells)
    return rows


def retrieve_context(example, cur):
    pages = {}
    evidences = []
    # Collect all page
    """
      meta types:
      - table_caption_18
      - cell_18_1_1
      - header_cell_18_0_0
      - sentence_0
      - item_4_25
      """
    tables = []
    for evidence in example["evidence"][:1]:
        content = evidence["content"]
        for item in content:
            page_id, meta = item.split("_", 1)
            if page_id not in pages:
                data = cur.execute("""
        SELECT data FROM wiki WHERE id = "{}"
        """.format(page_id))
                for item in data.fetchall():
                    pages[page_id] = json.loads(item[0])
            if meta.startswith("table_caption") or meta.startswith("cell") or meta.startswith("header_cell"):
                table_id = get_table_id(meta)
                if table_id in tables:
                    continue
                else:
                    tables.append(table_id)
                context = pages[page_id]["table_{}".format(table_id)]
                evidences.append(
                    {"table": process_table(context)})
            elif meta.startswith("item"):
                list_id = get_list_id(meta)
                for item in pages[page_id]["list_{}".format(list_id)]["list"]:
                    if item["id"] == meta:
                        context = item["value"]
                evidences.append(
                    {"item": clean_markup(context)})
            else:
                context = pages[page_id][meta]
                evidences.append(
                    {"sentence": clean_markup(context)})

    table_list, context_list = [], []
    for evidence in evidences:
        if "table" in evidence:
            table_list.append({
                "header": evidence["table"][0],
                "rows": evidence["table"][1:]})
        else:
            context_list.append(list(evidence.values())[0])
    return table_list, context_list


def is_table_involved(example):
    # Check if the example is involving table.
    # We only consider the first evidence
    for evidence in example["evidence"][:1]:  # list
        is_valid = False
        content = evidence["content"]
        evidence_type_count = defaultdict(int)
        for item in content:
            page_id, meta = item.split("_", 1)
            for evidence_type in EVIDENCE_TYPES:
                if meta.startswith(evidence_type):
                    evidence_type_count[evidence_type] += 1
        for evidence_type in evidence_type_count:
            if evidence_type in ["cell", "header_cell", "table_caption"]:
                is_valid = True
        if is_valid:
            return True
    return False


class FEVEROUS(datasets.GeneratorBasedBuilder):
    """The FEVEROUS dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "statement": datasets.Value("string"),
                    "table": datasets.features.Sequence(
                        {"header": datasets.features.Sequence(datasets.Value("string")),
                         "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))}
                    ),
                    "context": datasets.features.Sequence(datasets.Value("string")),
                    "label": datasets.Value("string"),
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
                gen_kwargs={"filepath": downloaded_files["train"],
                            "database": os.path.join(downloaded_files["database"], "feverous_wikiv1.db")}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"],
                            "database": os.path.join(downloaded_files["database"], "feverous_wikiv1.db")}),
        ]

    def _generate_examples(self, filepath, database):
        con = sqlite3.connect(database)
        cur = con.cursor()
        with open(filepath, "r") as f:
            count = -1
            for idx, line in enumerate(f):
                example = json.loads(line)
                statement = example["claim"]
                label = example["label"]
                # possible label: "NOT ENOUGH INFO", "REFUTES", "SUPPORTS"
                if is_table_involved(example):
                    # Retrieve related context from database
                    tables, contexts = retrieve_context(example, cur)
                    count += 1
                    yield count, {
                        "id": str(example["id"]),
                        "statement": statement,
                        "table": tables,
                        "context": contexts,
                        "label": label,
                    }
