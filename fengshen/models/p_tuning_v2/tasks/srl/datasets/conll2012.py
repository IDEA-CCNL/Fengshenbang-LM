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

_URL = "../../../data/CoNLL12/"
_TRAINING_FILE = "conll2012.train.txt"
_DEV_FILE = "conll2012.devel.txt"
_TEST_WSJ_FILE = "conll2012.test.txt"
# _TEST_BROWN_FILE = "conll.test.brown.txt"
CONLL12_LABELS = ['B-ARG0', 'B-ARGM-MNR', 'B-V', 'B-ARG1', 'B-ARG2', 'I-ARG2', 'O', 'I-ARG1', 'B-ARGM-ADV',
                  'B-ARGM-LOC', 'I-ARGM-LOC', 'I-ARG0', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-PRP',
                  'I-ARGM-PRP', 'B-ARGM-PRD', 'I-ARGM-PRD', 'B-R-ARGM-TMP', 'B-ARGM-DIR', 'I-ARGM-DIR',
                  'B-ARGM-DIS', 'B-ARGM-MOD', 'I-ARGM-ADV', 'I-ARGM-DIS', 'B-R-ARGM-LOC', 'B-ARG4',
                  'I-ARG4', 'B-R-ARG1', 'B-R-ARG0', 'I-R-ARG0', 'B-ARG3', 'B-ARGM-NEG', 'B-ARGM-CAU',
                  'I-ARGM-MNR', 'I-R-ARG1', 'B-C-ARG1', 'I-C-ARG1', 'B-ARGM-EXT', 'I-ARGM-EXT', 'I-ARGM-CAU',
                  'I-ARG3', 'B-C-ARGM-ADV', 'I-C-ARGM-ADV', 'B-ARGM-LVB', 'B-ARGM-REC', 'B-R-ARG3',
                  'B-R-ARG2', 'B-C-ARG0', 'I-C-ARG0', 'B-ARGM-ADJ', 'B-C-ARG2', 'I-C-ARG2', 'B-R-ARGM-CAU',
                  'B-R-ARGM-DIR', 'B-ARGM-GOL', 'I-ARGM-GOL', 'B-ARGM-DSP', 'I-ARGM-ADJ', 'I-R-ARG2',
                  'I-ARGM-NEG', 'B-ARGM-PRR', 'B-R-ARGM-ADV', 'I-R-ARGM-ADV', 'I-R-ARGM-LOC', 'B-ARGA',
                  'B-R-ARGM-MNR', 'I-R-ARGM-MNR', 'B-ARGM-COM', 'I-ARGM-COM', 'B-ARGM-PRX', 'I-ARGM-REC',
                  'B-R-ARG4', 'B-C-ARGM-LOC', 'I-C-ARGM-LOC', 'I-R-ARGM-DIR', 'I-ARGA', 'B-C-ARGM-TMP',
                  'I-C-ARGM-TMP', 'B-C-ARGM-CAU', 'I-C-ARGM-CAU', 'B-R-ARGM-PRD', 'I-R-ARGM-PRD',
                  'I-R-ARG3', 'B-C-ARG4', 'I-C-ARG4', 'B-ARGM-PNC', 'I-ARGM-PNC', 'B-ARG5', 'I-ARG5',
                  'B-C-ARGM-PRP', 'I-C-ARGM-PRP', 'B-C-ARGM-MNR', 'I-C-ARGM-MNR', 'I-R-ARGM-TMP',
                  'B-R-ARG5', 'I-ARGM-DSP', 'B-C-ARGM-DSP', 'I-C-ARGM-DSP', 'B-C-ARG3', 'I-C-ARG3',
                  'B-R-ARGM-COM', 'I-R-ARGM-COM', 'B-R-ARGM-PRP', 'I-R-ARGM-PRP', 'I-R-ARGM-CAU',
                  'B-R-ARGM-GOL', 'I-R-ARGM-GOL', 'B-R-ARGM-EXT', 'I-R-ARGM-EXT', 'I-R-ARG4',
                  'B-C-ARGM-EXT', 'I-C-ARGM-EXT', 'I-ARGM-MOD', 'B-C-ARGM-MOD', 'I-C-ARGM-MOD']


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
        Conll2005Config(name="conll2012", version=datasets.Version("1.0.0"), description="Conll2012 dataset"),
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
                            names=CONLL12_LABELS
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
            # "test_brown":  f"{_URL}{_TEST_BROWN_FILE}"
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name="validation", gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name="test_wsj", gen_kwargs={"filepath": downloaded_files["test_wsj"]}),
            # datasets.SplitGenerator(name="test_brown", gen_kwargs={"filepath": downloaded_files["test_brown"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:
                if line != '':
                    if line.split() == []:
                        continue
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