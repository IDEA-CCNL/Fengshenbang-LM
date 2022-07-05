#!/usr/bin/env bash
# -*- coding:utf-8 -*-

export PYTHONPATH="${PYTHONPATH}:./"

for data_format in entity relation event absa
do
    for dataset in $(ls converted_data/text2spotasoc/${data_format} | grep -v shot | grep -v ratio)
    do
        for seed in 1 2 3 4 5 6 7 8 9 10
        do
            rm -r converted_data/text2spotasoc/${data_format}/${dataset}_ratio/seed${seed} || true
            echo "Convert" converted_data/text2spotasoc/${data_format}/${dataset} "To" converted_data/text2spotasoc/${data_format}/${dataset}_ratio/seed${seed}
            python scripts/sample_data_ratio.py -seed ${seed} \
                -src converted_data/text2spotasoc/${data_format}/${dataset} \
                -tgt converted_data/text2spotasoc/${data_format}/${dataset}_ratio/seed${seed}
        done
    done
done


for data_format in entity relation event
do
    for dataset in $(ls converted_data/text2spotasoc/${data_format} | grep -v shot | grep -v ratio)
    do
        for seed in 1 2 3 4 5 6 7 8 9 10
        do
            rm -r converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed} || true
            echo "Convert" converted_data/text2spotasoc/${data_format}/${dataset} "To" converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed}
            python scripts/sample_data_shot.py -seed ${seed} \
                -src converted_data/text2spotasoc/${data_format}/${dataset} \
                -tgt converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed} \
                -task ${data_format}
        done
    done
done


for data_format in absa
do
    for dataset in $(ls converted_data/text2spotasoc/${data_format} | grep -v shot | grep -v ratio)
    do
        for seed in 1 2 3 4 5 6 7 8 9 10
        do
            rm -r converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed} || true
            echo "Convert" converted_data/text2spotasoc/${data_format}/${dataset} "To" converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed}
            python scripts/sample_data_shot.py -seed ${seed} \
                -src converted_data/text2spotasoc/${data_format}/${dataset} \
                -tgt converted_data/text2spotasoc/${data_format}/${dataset}_shot/seed${seed} \
                -task relation
        done
    done
done
