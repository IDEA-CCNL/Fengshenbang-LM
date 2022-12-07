import json
import datasets

_DESCRIPTION = """"""
_CITATION = """"""

_URL = "https://raw.githubusercontent.com/Ciaranshu/relogic/main/data/spider/raw/"
_TRAINING_FILE = "train.json"
_DEV_FILE = "dev.json"
_TEST_FILE = "test.json"

_URLS = {
    "train": f"{_URL}{_TRAINING_FILE}",
    "dev": f"{_URL}{_DEV_FILE}",
    "test": f"{_URL}{_TEST_FILE}",
}


class SQL2Text(datasets.GeneratorBasedBuilder):
    """SQL2Text dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/Ciaranshu/relogic",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            reader = json.load(f)
            for example_idx, example in enumerate(reader):
                yield example_idx, {
                    "question": example["question"],
                    "query": example["query"],
                }
