# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
"""

_DESCRIPTION = """\
The shared task of CoNLL-2003 concerns language-independent named entity recognition. We will concentrate on
four types of named entities: persons, locations, organizations and names of miscellaneous entities that do
not belong to the previous three groups.
The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on
a separate line and there is an empty line after each sentence. The first item on each line is a word, the second
a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags
and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only
if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag
B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Note the dataset uses IOB2
tagging scheme, whereas the original dataset uses IOB1.
For more details see https://www.clips.uantwerpen.be/conll2003/ner/ and https://www.aclweb.org/anthology/W03-0419
"""

_URL = "../../../data/CoNLL05/"
_TRAINING_FILE = "conll05.train.txt"
_DEV_FILE = "conll05.devel.txt"
_TEST_WSJ_FILE = "conll05.test.wsj.txt"
_TEST_BROWN_FILE = "conll05.test.brown.txt"


class Conll2005Config(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2005.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Conll2005Config, self).__init__(**kwargs)


class Conll2005(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        Conll2005Config(name="conll2005", version=datasets.Version("1.0.0"), description="Conll2005 dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "index": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['B-C-AM-TMP', 'B-C-AM-DIR', 'B-C-A2', 'B-R-AM-EXT', 'B-C-A0', 'I-AM-NEG', 'I-AM-ADV', 'B-C-V', 'B-C-AM-MNR', 'B-R-A3', 'I-AM-TM', 'B-V', 'B-R-A4', 'B-A5', 'I-A4', 'I-R-AM-LOC', 'I-C-A1', 'B-R-AA', 'I-C-A0', 'B-C-AM-EXT', 'I-C-AM-DIS', 'I-C-A5', 'B-A0', 'B-C-A4', 'B-C-AM-CAU', 'B-C-AM-NEG', 'B-AM-NEG', 'I-AM-MNR', 'I-R-A2', 'I-R-AM-TMP', 'B-AM', 'I-R-AM-PNC', 'B-AM-LOC', 'B-AM-REC', 'B-A2', 'I-AM-EXT', 'I-V', 'B-A3', 'B-A4', 'B-R-A0', 'I-AM-MOD', 'I-C-AM-CAU', 'B-R-AM-CAU', 'B-A1', 'B-R-AM-TMP', 'I-R-AM-EXT', 'B-C-AM-ADV', 'B-AM-ADV', 'B-R-A2', 'B-AM-CAU', 'B-R-AM-DIR', 'I-A5', 'B-C-AM-DIS', 'I-C-AM-MNR', 'B-AM-PNC', 'I-C-AM-LOC', 'I-R-A3', 'I-R-AM-ADV', 'I-A0', 'B-AM-EXT', 'B-R-AM-PNC', 'I-AM-DIS', 'I-AM-REC', 'B-C-AM-LOC', 'B-R-AM-ADV', 'I-AM', 'I-AM-CAU', 'I-AM-TMP', 'I-A1', 'I-C-A4', 'B-R-AM-LOC', 'I-C-A2', 'B-C-A5', 'O', 'B-R-AM-MNR', 'I-C-A3', 'I-R-AM-DIR', 'I-AM-PRD', 'B-AM-TM', 'I-A2', 'I-AA', 'I-AM-LOC', 'I-AM-PNC', 'B-AM-MOD', 'B-AM-DIR', 'B-R-A1', 'B-AM-TMP', 'B-AM-MNR', 'I-R-A0', 'B-AM-PRD', 'I-AM-DIR', 'B-AM-DIS', 'I-C-AM-ADV', 'I-R-A1', 'B-C-A3', 'I-R-AM-MNR', 'I-R-A4', 'I-C-AM-PNC', 'I-C-AM-TMP', 'I-C-V', 'I-A3', 'I-C-AM-EXT', 'B-C-A1', 'B-AA', 'I-C-AM-DIR', 'B-C-AM-PNC']
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test_wsj": f"{_URL}{_TEST_WSJ_FILE}",
            "test_brown":  f"{_URL}{_TEST_BROWN_FILE}"
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name="validation", gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name="test_wsj", gen_kwargs={"filepath": downloaded_files["test_wsj"]}),
            datasets.SplitGenerator(name="test_brown", gen_kwargs={"filepath": downloaded_files["test_brown"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:
                if line != '':
                    index = line.split()[0]

                    text = ' '.join(line.split()[1:]).strip()
                    tokens = text.split("|||")[0].split()
                    labels = text.split("|||")[1].split()
                    yield guid, {
                        "id": str(guid),
                        "index": index,
                        "tokens": tokens,
                        "tags": labels
                    }

                    guid += 1