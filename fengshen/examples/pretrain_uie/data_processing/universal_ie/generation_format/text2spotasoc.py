#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
from typing import List, Dict
from universal_ie.utils import tokens_to_str
from universal_ie.generation_format.generation_format import GenerationFormat, StructureMarker
from universal_ie.ie_format import Entity, Event, Label, Relation, Span


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


class Text2SpotAsoc(GenerationFormat):
    def __init__(self, structure_maker: StructureMarker, label_mapper: Dict = None, language: str = 'en') -> None:
        super().__init__(
            structure_maker=structure_maker,
            label_mapper=label_mapper,
            language=language
        )

    def annotate_entities(self, tokens: List[str], entities: List[Entity]):
        """ Convert Entities

        Args:
            tokens (List[str]): ['Trump', 'visits', 'China', '.']
            entities (List[Entity]): [description]

        Returns:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ] [ Geo-political : China ] }
        """
        return self.annonote_graph(tokens=tokens, entities=entities)[:2]

    def augment_source_span(self, tokens: List[str], span: Span):
        """[summary]

        Args:
            tokens (List[str]):
                ['Trump', 'visits', 'China', '.']
            span (Span):
                Trump

        Returns:
            [type]:
                ['(', 'Trump', ')', 'visits', 'China', '.']
        """
        return tokens[:span.indexes[0]] \
            + [self.structure_maker.source_span_start] \
            + tokens[span.indexes[0]:span.indexes[-1] + 1] \
            + [self.structure_maker.source_span_end] \
            + tokens[span.indexes[-1] + 1:]

    def annotate_given_entities(self, tokens: List[str], entities):
        """
        entityies is List
        :param tokens:
            ['Trump', 'visits', 'China', '.']
        :param entities:
            ['Trump', 'China']
        :return:
            source (str): ( Trump ) ( China ) : Trump visits China .
            target (str): { [ Person : Trump ] [ Geo-political : China ] }

        entityies is Entity
        :param tokens:
            ['Trump', 'visits', 'China', '.']
        :param entities:
            'Trump'
        :return:
            source (str): < Trump > visits China .
            target (str): { [ Person : Trump ] }
        """
        if isinstance(entities, list):
            entitytokens = []
            for entity in entities:
                entitytokens += [self.structure_maker.span_start]
                entitytokens += entity.span.tokens
                entitytokens += [self.structure_maker.span_end]
            source_text = tokens_to_str(
                entitytokens + [self.structure_maker.sep_marker] + tokens,
                language=self.language,
            )
            _, target_text = self.annonote_graph(tokens=tokens, entities=entities)[:2]

        elif isinstance(entities, Entity):
            marked_tokens = self.augment_source_span(tokens=tokens, span=entities.span)
            source_text = tokens_to_str(marked_tokens, language=self.language)
            _, target_text = self.annonote_graph(tokens=tokens, entities=[entities])[:2]

        return source_text, target_text

    def annotate_events(self, tokens: List[str], events: List[Event]):
        """
        :param tokens:
            ['Trump', 'visits', 'China', '.']
        :param events:

        :return:
            source (str): Trump visits China.
            target (str): { [ Visit : visits ( Person : Trump ) ( Location : China ) ] }
        """
        return self.annonote_graph(tokens=tokens, events=events)[:2]

    def annotate_event_given_predicate(self, tokens: List[str], event: Event):
        """Annotate Event Given Predicate

        Args:
            tokens (List[str]):
                ['Trump', 'visits', 'China', '.']
            event (Event): Given Predicate

        Returns:
            [type]: [description]
        """
        marked_tokens = self.augment_source_span(tokens=tokens, span=event.span)
        source_text = tokens_to_str(marked_tokens, language=self.language)
        _, target_text = self.annonote_graph(tokens=tokens, events=[event])[:2]
        return source_text, target_text

    def annotate_relation_extraction(self,
                                     tokens: List[str],
                                     relations: List[Relation]):
        """
        :param tokens:
            ['Trump', 'visits', 'China', '.']
        :param relations:

        :return:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ( Visit : China ) ] }
        """
        return self.annonote_graph(tokens=tokens, relations=relations)[:2]

    def annotate_entities_and_relation_extraction(self,
                                                  tokens: List[str],
                                                  entities: List[Entity],
                                                  relations: List[Relation]):
        """
        :param tokens:
            ['Trump', 'visits', 'China', '.']
        :param relations:

        :return:
            source (str): Trump visits China.
            target (str): { [ Person : Trump ( Visit : China ) ] [ Geo-political : China ] }
        """
        return self.annonote_graph(tokens=tokens, entities=entities, relations=relations)[:2]

    def annonote_graph(self,
                       tokens: List[str],
                       entities: List[Entity] = [],
                       relations: List[Relation] = [],
                       events: List[Event] = []):
        """Convert Entity Relation Event to Spot-Assocation Graph

        Args:
            tokens (List[str]): Token List
            entities (List[Entity], optional): Entity List. Defaults to [].
            relations (List[Relation], optional): Relation List. Defaults to [].
            events (List[Event], optional): Event List. Defaults to [].

        Returns:
            str: [description]
                {
                    [ Person : Trump ( Visit : China ) ]
                    [ Visit : visits ( Person : Trump ) ( Location : China ) ]
                    [ Geo-political : China ]
                }
            set: Set of Spot
            set: Set of Asoc
        """
        spot_dict = dict()
        asoc_dict = defaultdict(list)
        spot_str_rep_list = list()

        def add_spot(spot):
            spot_key = (tuple(spot.span.indexes), self.get_label_str(spot.label))
            spot_dict[spot_key] = spot

            if self.get_label_str(spot.label) not in self.record_role_map:
                self.record_role_map[self.get_label_str(spot.label)] = set()

        def add_asoc(spot, asoc: Label, tail):
            spot_key = (tuple(spot.span.indexes), self.get_label_str(spot.label))
            asoc_dict[spot_key] += [(tail.span.indexes, tail, self.get_label_str(asoc))]

            self.record_role_map[self.get_label_str(spot.label)].add(self.get_label_str(asoc))

        for entity in entities:
            add_spot(spot=entity)

        for relation in relations:
            add_spot(spot=relation.arg1)
            add_asoc(spot=relation.arg1, asoc=relation.label, tail=relation.arg2)

        for event in events:
            add_spot(spot=event)
            for arg_role, argument in event.args:
                add_asoc(spot=event, asoc=arg_role, tail=argument)

        spot_asoc_instance = list()
        for spot_key in sorted(spot_dict.keys()):
            offset, label = spot_key

            if spot_dict[spot_key].span.is_empty_span():
                continue

            spot_instance = {'span': spot_dict[spot_key].span.text,
                             'label': label,
                             'asoc': list(),
                             }
            for _, tail, asoc in sorted(asoc_dict.get(spot_key, [])):

                if tail.span.is_empty_span():
                    continue

                spot_instance['asoc'] += [(asoc, tail.span.text)]
            spot_asoc_instance += [spot_instance]

        target_text = convert_spot_asoc(
            spot_asoc_instance,
            structure_maker=self.structure_maker,
        )

        source_text = tokens_to_str(tokens, language=self.language)
        spot_labels = set([label for _, label in spot_dict.keys()])
        asoc_labels = set()
        for _, asoc_list in asoc_dict.items():
            for _, _, asoc in asoc_list:
                asoc_labels.add(asoc)
        return source_text, target_text, spot_labels, asoc_labels, spot_asoc_instance
