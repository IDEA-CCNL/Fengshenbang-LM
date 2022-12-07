import json

import datasets


_CITATION = """\
@article{xu2021grounding,
  title={Grounding Open-Domain Instructions to Automate Web Support Tasks},
  author={Xu, Nancy and Masling, Sam and Du, Michael and Campagna, Giovanni and Heck, Larry and Landay, James and Lam, Monica S},
  journal={arXiv preprint arXiv:2103.16057},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the RUSS.
"""

_HOMEPAGE = "https://github.com/xnancy/russ"
_LICENSE = ""

_URL = "https://raw.githubusercontent.com/talk2data/russ/master/data/"
_ALL_FILE = "eval_webtalk_official.tsv"

_URLS = {
    "train": f"{_URL}{_ALL_FILE}",
    "dev": f"{_URL}{_ALL_FILE}",
    "test": f"{_URL}{_ALL_FILE}",
}

LIST = {
    "train": (0, 486),
    "dev": (486, 586),
    "test": (586, 739),
}


class RUSS(datasets.GeneratorBasedBuilder):
    """The RUSS dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
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
                gen_kwargs={"filepath": downloaded_files["train"], "fold": "train"}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"], "fold": "dev"}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"], "fold": "test"}),
        ]

    def _generate_examples(self, filepath, fold):
        """Yields examples."""
        """We need to split the file into train, dev, and test."""
        parses = []
        with open(filepath, "r") as f:
            for line in f:
                uid, instruction, parse = line.strip().split("\t")
                parses.append({
                    "uid": uid,
                    "instruction": instruction,
                    "parse": parse
                })
        count = 0
        for example in parses[LIST[fold][0]: LIST[fold][1]]:
            count += 1
            yield count, {
                "id": example["uid"],
                "question": example["instruction"],
                "query": example["parse"]
            }


