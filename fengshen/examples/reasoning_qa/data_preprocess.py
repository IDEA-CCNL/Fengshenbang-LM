import json
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_json(cls, input_file):
    """Reads a JSON file."""
    with open(input_file, "r") as f:
      return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file):
    """Reads a JSON Lines file."""
    with open(input_file, "r") as f:
        res = []
        for ln in tqdm(f.readlines()):
            if ln:
                res.append(json.loads(ln))
    return res
        #  return [json.loads(ln) for ln in f.readlines() if ln]
        #  for idx, ln in enumerate(f.readlines()):
            #  if ln:
                #  try:
                    #  a = json.loads(ln)
                #  except:
                    #  print(idx, ln)

  @classmethod
  def _read_txt(cls, input_file):
    """Reads a txt file."""
    with open(input_file, "r") as f:
      return f.readlines()

