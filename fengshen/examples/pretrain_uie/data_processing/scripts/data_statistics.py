import json
import os
import sys
from collections import Counter
import tabulate


def count_line_in_file(filename):
    return sum([1 for _ in open(filename)])


def count_record_in_file(filename, key):
    counter = Counter()
    for line in open(filename):
        instance = json.loads(line)
        counter.update([key + ' entity'] * len(instance['entity']))
        counter.update([key + ' relation'] * len(instance['relation']))
        counter.update([key + ' event'] * len(instance['event']))
        for event in instance['event']:
            counter.update([key + ' role'] * len(event['args']))
    return counter


def count_folder(folder_name):
    data_map = {
        'train': 'train.json',
        'val': 'val.json',
        'test': 'test.json',
    }
    intance_counter = {'name': folder_name}
    for key, name in data_map.items():
        filename = f"{folder_name}/{name}"
        if not os.path.exists(filename):
            sys.stderr.write(f'[warn] {filename} not exists.\n')
            continue
        intance_counter[key] = count_line_in_file(filename)
        intance_counter.update(count_record_in_file(filename, key))

    for key in ['entity', 'relation', 'event']:
        filename = f"{folder_name}/{key}.schema"
        if not os.path.exists(filename):
            sys.stderr.write(f'[warn] {filename} not exists.\n')
            intance_counter[key] = 0
            continue
        intance_counter[key] = len(json.loads(open(filename).readline()))

    return intance_counter


def walk_dir(folder_name):

    for root, dirs, files in os.walk(folder_name):
        for file in dirs:
            folder_name = os.path.join(root, file)
            if os.path.exists(f"{os.path.join(root, file)}/record.schema"):
                yield os.path.join(root, file)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data')
    parser.add_argument('-f', dest='format', default='simple')
    options = parser.parse_args()

    folder_list = list()

    for folder_name in walk_dir(options.data):
        if 'shot' in folder_name or 'ratio' in folder_name:
            continue
        folder_list += [count_folder(folder_name)]

    col_name = ['name',
                'entity', 'relation', 'event',
                'train', 'val', 'test',
                'train entity', 'train relation', 'train event', 'train role',
                'val entity', 'val relation', 'val event', 'val role',
                'test entity', 'test relation', 'test event', 'test role',
                ]
    table = []
    for data_info in folder_list:
        row = [data_info.get(col, 0) for col in col_name]
        table += [row]
    table.sort()
    print(
        tabulate.tabulate(
            tabular_data=table,
            headers=col_name,
            tablefmt=options.format,
        )
    )


if __name__ == "__main__":
    main()
