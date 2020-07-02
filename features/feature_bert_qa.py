#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tokenizers import BertWordPieceTokenizer

from common.text import get_valid_windows
from .feature_qa import QaDataBuilder


class BertQaDataBuilder(QaDataBuilder):

    def __init__(self, config):
        super().__init__(config)
        self.preprocessor = BertWordPieceTokenizer(config.vocab_file,
                                                   lowercase=config.lower_case)
        self.cls_id = self.preprocessor.token_to_id('[CLS]')
        self.sep_id = self.preprocessor.token_to_id('[SEP]')
        self.max_seq_length = config.max_seq_length
        self.max_ctx_tokens = 0  # updated in input_to_feature

    def get_max_ctx_tokens(self, q_len):
        return self.max_seq_length - q_len - 3   # 1 [CLS], 2 [SEP]

    def get_ctx_offset(self, q_len):
        return q_len + 2  # +2 for [CLS], [SEP] since q is before ctx

    def input_to_feature(self, one_input):
        if len(one_input) == 3:
            qid, question, context = one_input
            label = None
        elif len(one_input) == 4:
            qid, question, context, label = one_input
        else:
            raise ValueError('number of inputs not valid error: {}'.format(
                len(one_input)))

        q_codes = self.preprocessor.encode(question,
                                           add_special_tokens=False)
        ctx_codes = self.preprocessor.encode(context,
                                             add_special_tokens=False)
        q_ids = q_codes.ids
        ctx_ids = ctx_codes.ids
        ctx_tokens = ctx_codes.tokens
        ctx_spans = ctx_codes.offsets

        label_info = self.may_process_label(label, (context, ctx_codes))
        ans_cls, ans_start, ans_end = label_info
        feature_dict = {
            'feature_id': qid, 'question': question, 'context': context,
            'question_tokens': q_codes.tokens,
            'label': label, 'cls': ans_cls,
            'answer_start': ans_start, 'answer_end': ans_end,
        }
        q_len = len(q_codes.tokens)
        ctx_token_len = len(ctx_codes.tokens)
        max_ctx_tokens = self.get_max_ctx_tokens(q_len)
        context_valid_spans = get_valid_windows(ctx_token_len, max_ctx_tokens,
                                                self.config.context_stride)
        win_offset = self.get_ctx_offset(q_len)
        for win_span in context_valid_spans:
            win_start, win_end = win_span
            win_ctx_ids = ctx_ids[win_start:win_end]
            feature_dict = self.build_ids(feature_dict, q_ids, win_ctx_ids)
            win_ctx_tokens = ctx_tokens[win_start:win_end]
            win_ctx_spans = ctx_spans[win_start:win_end]

            cls, answer_start, answer_end = self.adjust_label(
                feature_dict, win_offset, win_span)
            if feature_dict['label'] is not None and cls is None:
                # has label, but no valid answer_span in current window
                continue
            self.num_examples += 1
            feature_id = '{}_{}'.format(self.num_examples,
                                        feature_dict['feature_id'])
            feature_dict['feature_id'] = feature_id
            feature_dict['context_tokens'] = win_ctx_tokens
            feature_dict['context_spans'] = win_ctx_spans
            feature_dict['answer_start'] = answer_start
            feature_dict['answer_end'] = answer_end
            feature = self.feature(**feature_dict)
            yield feature

    def build_ids(self, feature_dict, q_ids, win_ctx_ids):
        # for BERT, first put cls, then put q and ctx
        first_part_ids = [self.cls_id] + q_ids + [self.sep_id]
        second_part_ids = win_ctx_ids + [self.sep_id]
        input_ids = first_part_ids + second_part_ids

        segment_ids = [0] * len(first_part_ids) + [1] * len(second_part_ids)
        # pad to max_seq_length
        input_len = len(input_ids)
        input_ids += [0] * (self.max_seq_length - input_len)
        segment_ids += [0] * (self.max_seq_length - input_len)

        feature_dict['input_ids'] = input_ids
        feature_dict['segment_ids'] = segment_ids
        return feature_dict

    @staticmethod
    def two_seq_str_fn(feat):
        q_str = ['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            'q_idx', 'token', 'input_idx', 'input_id')]
        q_str.extend(['|{:>5}|{:>15}|{:>10}|{:>10}'.format(
            q_idx, q_token, q_idx + 1, feat.input_ids[q_idx + 1])
            for q_idx, q_token in enumerate(feat.question_tokens)])

        ctx_str = ['|{:>5}|{:>15}|{:>15}|{:>10}|{:>10}'.format(
            'c_idx', 'token', 'span', 'input_idx', 'input_id')]
        q_len = len(feat.question_tokens)
        ctx_str.extend(['|{:>5}|{:>15}|{:>15}|{:>10}|{:>10}'.format(
            c_idx, c_token, str(feat.context_spans[c_idx]),
            c_idx + 2 + q_len, feat.input_ids[c_idx + 2 + q_len])
            for c_idx, c_token in enumerate(feat.context_tokens)])

        return q_str, ctx_str
