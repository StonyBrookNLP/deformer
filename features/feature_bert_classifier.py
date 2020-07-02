#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tokenizers import BertWordPieceTokenizer

from .feature_classifier import ClassifierDataBuilder


class BertClassifierDataBuilder(ClassifierDataBuilder):

    def __init__(self, config):
        super().__init__(config)
        self.preprocessor = BertWordPieceTokenizer(config.vocab_file,
                                                   lowercase=config.lower_case)
        self.cls_id = self.preprocessor.token_to_id('[CLS]')
        self.sep_id = self.preprocessor.token_to_id('[SEP]')
        self.max_seq_length = config.max_seq_length

    def build_one_input_ids(self, seq1_codes, seq2_codes):
        seq1_ids = seq1_codes.ids
        seq2_ids = seq2_codes.ids
        seq1_tokens = seq1_codes.tokens
        seq2_tokens = seq2_codes.tokens
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer
        # sequence one token at a time. This makes more sense than
        # truncating an equal percent of tokens from each, since if one
        # sequence is very short then each token that's truncated likely
        # contains more information than a longer sequence.
        while True:
            total_length = len(seq1_tokens) + len(seq2_tokens)
            if total_length <= self.max_seq_length - 3:
                # Account for [CLS], [SEP], [SEP] with "- 3"
                # logger.info('truncation finished.')
                break
            if len(seq1_tokens) > len(seq2_tokens):
                seq1_tokens.pop()
                seq1_ids.pop()
            else:
                seq2_tokens.pop()
                seq2_ids.pop()

        first_part_ids = [self.cls_id] + seq1_ids + [self.sep_id]
        second_part_ids = seq2_ids + [self.sep_id]
        input_ids = first_part_ids + second_part_ids
        segment_ids = [0] * len(first_part_ids) + [1] * len(second_part_ids)
        # pad to max_seq_length
        input_len = len(input_ids)
        input_ids += [0] * (self.max_seq_length - input_len)
        segment_ids += [0] * (self.max_seq_length - input_len)
        return input_ids, segment_ids, seq1_tokens, seq2_tokens

    def build_ids(self, seq1_list, seq2_codes):
        part1_ids = []
        part2_ids = []
        seq1_tokens = []
        seq2_tokens = []
        for s1 in seq1_list:
            s1_codes = self.preprocessor.encode(s1, add_special_tokens=False)
            one_output = self.build_one_input_ids(s1_codes, seq2_codes)
            p1_ids, sp2_ids, s1_tokens, s2_tokens = one_output
            part1_ids.extend(p1_ids)
            part2_ids.extend(sp2_ids)
            seq1_tokens.extend(s1_tokens)
            seq2_tokens.extend(s2_tokens)
        return part1_ids, part2_ids, seq1_tokens, seq2_tokens

    def set_ids(self, feature_dict, one_output):
        input_ids, segment_ids, seq1_tokens, seq2_tokens = one_output
        feature_dict['input_ids'] = input_ids
        feature_dict['segment_ids'] = segment_ids
        feature_dict['seq1_tokens'] = seq1_tokens
        feature_dict['seq2_tokens'] = seq2_tokens
        return feature_dict

    def input_to_feature(self, one_input):
        if len(one_input) == 3:
            eid, seq1, seq2 = one_input
            label = None
        elif len(one_input) == 4:
            eid, seq1, seq2, label = one_input
        else:
            raise ValueError('number of inputs not valid error: {}'.format(
                len(one_input)))

        seq2_codes = self.preprocessor.encode(seq2, add_special_tokens=False)
        ans_cls = self.may_process_label(label, None)
        feature_dict = {
            'feature_id': eid, 'label': label, 'cls': ans_cls,
            'seq1': seq1, 'seq2': seq2,
        }
        if isinstance(seq1, list):  # for race multiple choice
            seq1_list = seq1
        else:
            seq1_list = [seq1]  # for boolq, mnli, qqp
        one_output = self.build_ids(seq1_list, seq2_codes)
        feature_dict = self.set_ids(feature_dict, one_output)
        self.num_examples += 1
        feature_id = '{}_{}'.format(self.num_examples,
                                    feature_dict['feature_id'])
        feature_dict['feature_id'] = feature_id
        feature = self.feature(**feature_dict)
        yield feature

    @staticmethod
    def two_seq_str_fn(feat):
        seq1_str = ['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            'seq1_idx', 'token', 'input_idx', 'input_id')]
        seq1_str.extend(['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            q_idx, q_token, q_idx + 1, feat.input_ids[q_idx + 1])
            for q_idx, q_token in enumerate(feat.seq1_tokens)])

        seq2_str = ['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            'seq2_idx', 'token', 'input_idx', 'input_id')]
        seq1_len = len(feat.seq1_tokens)
        seq2_str.extend(['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            c_idx, c_token, c_idx + 2 + seq1_len,
            feat.input_ids[c_idx + 2 + seq1_len])
            for c_idx, c_token in enumerate(feat.seq2_tokens)])

        return seq1_str, seq2_str
