#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import os
import json
from typing import Dict, List
from tqdm import tqdm
from universal_ie.generation_format.generation_format import GenerationFormat
from universal_ie.generation_format import generation_format_dict
from universal_ie.generation_format.structure_marker import BaseStructureMarker
from universal_ie.dataset import Dataset
from universal_ie.ie_format import Sentence


def convert_graph(
    generation_class: GenerationFormat,
    output_folder: str,
    datasets: Dict[str, List[Sentence]],
    language: str = "en",
    label_mapper: Dict = None,
):
    convertor = generation_class(
        structure_maker=BaseStructureMarker(),
        language=language,
        label_mapper=label_mapper,
    )

    counter = Counter()

    os.makedirs(output_folder, exist_ok=True)

    schema_counter = {
        "entity": list(),
        "relation": list(),
        "event": list(),
    }
    for data_type, instance_list in datasets.items():
        with open(os.path.join(output_folder, f"{data_type}.json"), "w") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                converted_graph = convertor.annonote_graph(
                    tokens=instance.tokens,
                    entities=instance.entities,
                    relations=instance.relations,
                    events=instance.events,
                )
                # breakpoint()
                src, tgt, spot_labels, asoc_labels = converted_graph[:4]
                spot_asoc = converted_graph[4]

                schema_counter["entity"] += instance.entities
                schema_counter["relation"] += instance.relations
                schema_counter["event"] += instance.events

                output.write(
                    "%s\n"
                    % json.dumps(
                        {
                            "text": src,
                            "tokens": instance.tokens,
                            "record": tgt,
                            "entity": [
                                entity.to_offset(label_mapper)
                                for entity in instance.entities # [[[we]([3])ORG]]
                            ],
                            "relation": [
                                relation.to_offset(
                                    ent_label_mapper=label_mapper,
                                    rel_label_mapper=label_mapper,
                                )
                                for relation in instance.relations
                            ],
                            "event": [
                                event.to_offset(evt_label_mapper=label_mapper)
                                for event in instance.events
                            ],
                            "spot": list(spot_labels),
                            "asoc": list(asoc_labels),
                            "spot_asoc": spot_asoc,
                        },
                        ensure_ascii=False,
                    )
                )
    convertor.output_schema(os.path.join(output_folder, "record.schema"))
    convertor.get_entity_schema(schema_counter["entity"]).write_to_file(
        os.path.join(output_folder, f"entity.schema")
    )
    convertor.get_relation_schema(schema_counter["relation"]).write_to_file(
        os.path.join(output_folder, f"relation.schema")
    )
    convertor.get_event_schema(schema_counter["event"]).write_to_file(
        os.path.join(output_folder, f"event.schema")
    )
    print(output_folder, counter)
    print("==========================")
   






def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-format", dest="generation_format", default="spotasoc")
    parser.add_argument("-config", dest="config", default="data_config/relation")
    parser.add_argument("-output", dest="output", default="relation")
    options = parser.parse_args()

    generation_class = generation_format_dict.get(options.generation_format)

    if os.path.isfile(options.config):
        config_list = [options.config]
    else:
        config_list = [
            os.path.join(options.config, x) for x in os.listdir(options.config)
        ]
    # breakpoint()
    # 读取每一个data_config之下的文件的yaml，实例化一个dataset的对象
    for filename in config_list:
        dataset = Dataset.load_yaml_file(filename)

        datasets = dataset.load_dataset()
        label_mapper = dataset.mapper
        # print(label_mapper)

        output_name = (
            f"converted_data/text2{options.generation_format}/{options.output}/"
            + dataset.name
        )

      
        convert_graph(  generation_class,  output_name,  datasets=datasets, language=dataset.language,label_mapper=label_mapper)



if __name__ == "__main__":
    main()
