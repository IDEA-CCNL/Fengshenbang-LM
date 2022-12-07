"""A large crowd-sourced dataset for developing natural language interfaces for relational databases"""
# TODO： This code can be push to HuggingFace as a new contribution.
"""
The Query and DBEngine class are adopted from the WikiSQL official GitHub page.
https://github.com/salesforce/WikiSQL/tree/master/lib
"""

"""https://github.com/salesforce/WikiSQL/blob/master/lib/common.py"""


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()


"""https://github.com/salesforce/WikiSQL/blob/master/lib/query.py"""
import re
from copy import deepcopy

re_whitespace = re.compile(r'\s+', flags=re.UNICODE)


class Query:
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG',
            'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, conditions=tuple(), ordered=False):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.conditions = list(conditions)
        self.ordered = ordered

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            indices = self.sel_index == other.sel_index and self.agg_index == other.agg_index
            if other.ordered:
                conds = [(col, op, str(cond).lower()) for col, op, cond in self.conditions] == [
                    (col, op, str(cond).lower()) for col, op, cond in other.conditions]
            else:
                conds = set([(col, op, str(cond).lower()) for col, op, cond in self.conditions]) == set(
                    [(col, op, str(cond).lower()) for col, op, cond in other.conditions])

            return indices and conds
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops[self.agg_index],
            sel='col{}'.format(self.sel_index),
        )
        if self.conditions:
            rep += ' WHERE ' + ' AND '.join(
                ['{} {} {}'.format('col{}'.format(i), self.cond_ops[o], v) for i, o, v in self.conditions])
        return rep

    def to_dict(self):
        return {'sel': self.sel_index, 'agg': self.agg_index, 'conds': self.conditions}

    def lower(self):
        conds = []
        for col, op, cond in self.conditions:
            conds.append([col, op, cond.lower()])
        return self.__class__(self.sel_index, self.agg_index, conds)

    @classmethod
    def from_dict(cls, d, ordered=False):
        return cls(sel_index=d['sel'], agg_index=d['agg'], conditions=d['conds'], ordered=ordered)

    @classmethod
    def from_tokenized_dict(cls, d):
        conds = []
        for col, op, val in d['conds']:
            conds.append([col, op, detokenize(val)])
        return cls(d['sel'], d['agg'], conds)

    @classmethod
    def from_generated_dict(cls, d):
        conds = []
        for col, op, val in d['conds']:
            end = len(val['words'])
            conds.append([col, op, detokenize(val)])
        return cls(d['sel'], d['agg'], conds)

    @classmethod
    def from_sequence(cls, sequence, table, lowercase=True):
        sequence = deepcopy(sequence)
        if 'symend' in sequence['words']:
            end = sequence['words'].index('symend')
            for k, v in sequence.items():
                sequence[k] = v[:end]
        terms = [{'gloss': g, 'word': w, 'after': a} for g, w, a in
                 zip(sequence['gloss'], sequence['words'], sequence['after'])]
        headers = [detokenize(h) for h in table['header']]

        # lowercase everything and truncate sequence
        if lowercase:
            headers = [h.lower() for h in headers]
            for i, t in enumerate(terms):
                for k, v in t.items():
                    t[k] = v.lower()
        headers_no_whitespcae = [re.sub(re_whitespace, '', h) for h in headers]

        # get select
        if 'symselect' != terms.pop(0)['word']:
            raise Exception('Missing symselect operator')

        # get aggregation
        if 'symagg' != terms.pop(0)['word']:
            raise Exception('Missing symagg operator')
        agg_op = terms.pop(0)['word']

        if agg_op == 'symcol':
            agg_op = ''
        else:
            if 'symcol' != terms.pop(0)['word']:
                raise Exception('Missing aggregation column')
        try:
            agg_op = cls.agg_ops.index(agg_op.upper())
        except Exception as e:
            raise Exception('Invalid agg op {}'.format(agg_op))

        def find_column(name):
            return headers_no_whitespcae.index(re.sub(re_whitespace, '', name))

        def flatten(tokens):
            ret = {'words': [], 'after': [], 'gloss': []}
            for t in tokens:
                ret['words'].append(t['word'])
                ret['after'].append(t['after'])
                ret['gloss'].append(t['gloss'])
            return ret

        where_index = [i for i, t in enumerate(terms) if t['word'] == 'symwhere']
        where_index = where_index[0] if where_index else len(terms)
        flat = flatten(terms[:where_index])
        try:
            agg_col = find_column(detokenize(flat))
        except Exception as e:
            raise Exception('Cannot find aggregation column {}'.format(flat['words']))
        where_terms = terms[where_index + 1:]

        # get conditions
        conditions = []
        while where_terms:
            t = where_terms.pop(0)
            flat = flatten(where_terms)
            if t['word'] != 'symcol':
                raise Exception('Missing conditional column {}'.format(flat['words']))
            try:
                op_index = flat['words'].index('symop')
                col_tokens = flatten(where_terms[:op_index])
            except Exception as e:
                raise Exception('Missing conditional operator {}'.format(flat['words']))
            cond_op = where_terms[op_index + 1]['word']
            try:
                cond_op = cls.cond_ops.index(cond_op.upper())
            except Exception as e:
                raise Exception('Invalid cond op {}'.format(cond_op))
            try:
                cond_col = find_column(detokenize(col_tokens))
            except Exception as e:
                raise Exception('Cannot find conditional column {}'.format(col_tokens['words']))
            try:
                val_index = flat['words'].index('symcond')
            except Exception as e:
                raise Exception('Cannot find conditional value {}'.format(flat['words']))

            where_terms = where_terms[val_index + 1:]
            flat = flatten(where_terms)
            val_end_index = flat['words'].index('symand') if 'symand' in flat['words'] else len(where_terms)
            cond_val = detokenize(flatten(where_terms[:val_end_index]))
            conditions.append([cond_col, cond_op, cond_val])
            where_terms = where_terms[val_end_index + 1:]
        q = cls(agg_col, agg_op, conditions)
        return q

    @classmethod
    def from_partial_sequence(cls, agg_col, agg_op, sequence, table, lowercase=True):
        sequence = deepcopy(sequence)
        if 'symend' in sequence['words']:
            end = sequence['words'].index('symend')
            for k, v in sequence.items():
                sequence[k] = v[:end]
        terms = [{'gloss': g, 'word': w, 'after': a} for g, w, a in
                 zip(sequence['gloss'], sequence['words'], sequence['after'])]
        headers = [detokenize(h) for h in table['header']]

        # lowercase everything and truncate sequence
        if lowercase:
            headers = [h.lower() for h in headers]
            for i, t in enumerate(terms):
                for k, v in t.items():
                    t[k] = v.lower()
        headers_no_whitespcae = [re.sub(re_whitespace, '', h) for h in headers]

        def find_column(name):
            return headers_no_whitespcae.index(re.sub(re_whitespace, '', name))

        def flatten(tokens):
            ret = {'words': [], 'after': [], 'gloss': []}
            for t in tokens:
                ret['words'].append(t['word'])
                ret['after'].append(t['after'])
                ret['gloss'].append(t['gloss'])
            return ret

        where_index = [i for i, t in enumerate(terms) if t['word'] == 'symwhere']
        where_index = where_index[0] if where_index else len(terms)
        where_terms = terms[where_index + 1:]

        # get conditions
        conditions = []
        while where_terms:
            t = where_terms.pop(0)
            flat = flatten(where_terms)
            if t['word'] != 'symcol':
                raise Exception('Missing conditional column {}'.format(flat['words']))
            try:
                op_index = flat['words'].index('symop')
                col_tokens = flatten(where_terms[:op_index])
            except Exception as e:
                raise Exception('Missing conditional operator {}'.format(flat['words']))
            cond_op = where_terms[op_index + 1]['word']
            try:
                cond_op = cls.cond_ops.index(cond_op.upper())
            except Exception as e:
                raise Exception('Invalid cond op {}'.format(cond_op))
            try:
                cond_col = find_column(detokenize(col_tokens))
            except Exception as e:
                raise Exception('Cannot find conditional column {}'.format(col_tokens['words']))
            try:
                val_index = flat['words'].index('symcond')
            except Exception as e:
                raise Exception('Cannot find conditional value {}'.format(flat['words']))

            where_terms = where_terms[val_index + 1:]
            flat = flatten(where_terms)
            val_end_index = flat['words'].index('symand') if 'symand' in flat['words'] else len(where_terms)
            cond_val = detokenize(flatten(where_terms[:val_end_index]))
            conditions.append([cond_col, cond_op, cond_val])
            where_terms = where_terms[val_end_index + 1:]
        q = cls(agg_col, agg_op, conditions)
        return q


"""https://github.com/salesforce/WikiSQL/blob/master/lib/dbengine.py"""
import re

import records
from babel.numbers import parse_decimal, NumberFormatError

schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


class DBEngine:
    """
    I changed the code "val = float(parse_decimal(val))"
    to "val = float(parse_decimal(val, locale='en_US'))"
    to prevent bugs due to OS and file encoding.
    (for more details why i did that please review the source code of babel package)
    """

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        """
        My lesson is, to make this line perform normally, you must keep an older version of sqlalchemy, 1.3.10 for example.
        """
        table_info = self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = Query.agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and isinstance(val, str):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val, locale='en_US'))
                    # val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, Query.cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        out = self.conn.query(query, **where_map)
        return [o.result for o in out]


import json
import os

import datasets

_CITATION = """\
@article{zhongSeq2SQL2017,
  author    = {Victor Zhong and
               Caiming Xiong and
               Richard Socher},
  title     = {Seq2SQL: Generating Structured Queries from Natural Language using
               Reinforcement Learning},
  journal   = {CoRR},
  volume    = {abs/1709.00103},
  year      = {2017}
}
"""

_DESCRIPTION = """\
A large crowd-sourced dataset for developing natural language interfaces for relational databases
"""

_DATA_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]


class WikiSQL(datasets.GeneratorBasedBuilder):
    """WikiSQL: A large crowd-sourced dataset for developing natural language interfaces for relational databases"""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "phase": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "table": {
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "page_title": datasets.Value("string"),
                        "page_id": datasets.Value("string"),
                        "types": datasets.features.Sequence(datasets.Value("string")),
                        "id": datasets.Value("string"),
                        "section_title": datasets.Value("string"),
                        "caption": datasets.Value("string"),
                        "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                        "name": datasets.Value("string"),
                    },
                    "sql": {
                        "human_readable": datasets.Value("string"),
                        "sel": datasets.Value("int32"),
                        "agg": datasets.Value("int32"),
                        "conds": datasets.features.Sequence(
                            {
                                "column_index": datasets.Value("int32"),
                                "operator_index": datasets.Value("int32"),
                                "condition": datasets.Value("string"),
                            }
                        ),
                    },
                    "answer_text": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/salesforce/WikiSQL",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_DATA_URL)
        dl_dir = os.path.join(dl_dir, "data")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "main_filepath": os.path.join(dl_dir, "test.jsonl"),
                    "tables_filepath": os.path.join(dl_dir, "test.tables.jsonl"),
                    "db_filepath": os.path.join(dl_dir, 'test.db')
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "main_filepath": os.path.join(dl_dir, "dev.jsonl"),
                    "tables_filepath": os.path.join(dl_dir, "dev.tables.jsonl"),
                    "db_filepath": os.path.join(dl_dir, 'dev.db')
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "main_filepath": os.path.join(dl_dir, "train.jsonl"),
                    "tables_filepath": os.path.join(dl_dir, "train.tables.jsonl"),
                    "db_filepath": os.path.join(dl_dir, 'train.db')
                },
            ),
        ]

    def _convert_to_human_readable(self, sel, agg, columns, conditions):
        """Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""

        rep = "SELECT {agg} {sel} FROM table".format(
            agg=_AGG_OPS[agg], sel=columns[sel] if columns is not None else "col{}".format(sel)
        )

        if conditions:
            rep += " WHERE " + " AND ".join(["{} {} {}".format(columns[i], _COND_OPS[o], v) for i, o, v in conditions])
        return " ".join(rep.split())

    def _generate_examples(self, main_filepath, tables_filepath, db_filepath):
        """Yields examples."""

        # Build dictionary to table_ids:tables
        with open(tables_filepath, encoding="utf-8") as f:
            tables = [json.loads(line) for line in f]
            id_to_tables = {x["id"]: x for x in tables}

        # Build the DBEngine
        db_engine = DBEngine(db_filepath)

        with open(main_filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                row["table"] = id_to_tables[row["table_id"]]
                del row["table_id"]

                # Get the result of the query.
                row['answer_text'] = [str(result) for result in db_engine.execute_query(row["table"]["id"], Query.from_dict(row["sql"]))]

                # Handle missing data
                row["table"]["page_title"] = row["table"].get("page_title", "")
                row["table"]["section_title"] = row["table"].get("section_title", "")
                row["table"]["caption"] = row["table"].get("caption", "")
                row["table"]["name"] = row["table"].get("name", "")
                row["table"]["page_id"] = str(row["table"].get("page_id", ""))

                # Fix row types
                row["table"]["rows"] = [[str(e) for e in r] for r in row["table"]["rows"]]

                # Get human-readable version
                row["sql"]["human_readable"] = self._convert_to_human_readable(
                    row["sql"]["sel"],
                    row["sql"]["agg"],
                    row["table"]["header"],
                    row["sql"]["conds"],
                )

                # Restructure sql->conds
                # - wikiSQL provides a tuple [column_index, operator_index, condition]
                #   as 'condition' can have 2 types (float or str) we convert to dict
                for i in range(len(row["sql"]["conds"])):
                    row["sql"]["conds"][i] = {
                        "column_index": row["sql"]["conds"][i][0],
                        "operator_index": row["sql"]["conds"][i][1],
                        "condition": str(row["sql"]["conds"][i][2]),
                    }

                yield idx, row
