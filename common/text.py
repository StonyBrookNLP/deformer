#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

# Stop words modified from
# https://github.com/explosion/spaCy/blob/23025d3b05572a840ec91301092f8bee68cb1753/spacy/lang/en/stop_words.py

# STOP_WORDS = set("""
# '' 'd 'll 'm 're 's 've + ^ ``
# a about above across after afterwards again against ain all almost
# already also although always am among amongst an and another any
# anyhow anyone anything anyway anywhere aren as at
# be because been before being below between both but by can couldn
# d did didn do does doesn doing don done down during
# each even every everywhere except few for from further get give go
# had hadn has hasn have haven having he her here hereafter hereby herein
# hereupon hers herself him himself his how however
# i if in indeed into is isn it its itself just latter latterly less ll
# m ma many may me meanwhile mightn mine more moreover most mostly much must
# mustn my myself n't namely needn neither never nevertheless no none noone
# nor not nothing now nowhere o of off often on once only or other others
# otherwise our ours ourselves out over own per perhaps please
# rather re really regarding s same say see seem seemed seeming seems shan
# she should shouldn so some somehow someone something sometime sometimes
# somewhere still such t than that the their theirs them themselves then thence
# there thereafter thereby therefore therein thereupon these they this those
# though through throughout thru thus to too toward towards
# under unless until up upon us used using ve very via
# was wasn we well were weren what whatever when whence whenever
# where whereafter whereas whereby wherein whereupon wherever whether which
# while whither who whoever whole whom whose why will with within without
# would wouldn yet y you your yours yourself yourselves
# """.split())

NULL = '<null>'
UNK = '<unk>'


# def rm_white_space(text):
#     return ' '.join(rm_special_chars(text).split())


# def rm_special_chars(text):
#     return re.sub(r'[' + re.escape(r"&*\^`{|}~") + ']', '', text)


def word_tokenize(text):
    """Split on whitespace and punctuation."""
    return re.findall(r'\w+|[^\w\s]', text, re.U)


def byte_to_char_offset(unicode_str):
    """map unicode_str byte offsets to character offsets.
    """
    offset_map = {}
    byte_offset = 0
    for char_idx, char in enumerate(unicode_str):
        offset_map[byte_offset] = char_idx
        byte_offset += len(char.encode('utf-8'))

    offset_map[byte_offset] = len(unicode_str)
    return offset_map


# def normalize(text):
#     """Resolve different type of unicode encodings."""
#     return unicodedata.normalize('NFD', text)


# def tokenize_text_word_piece(input_text, lower_case, vocab, is_context=False):
#     input_text = input_text.lower() if lower_case else input_text
#     if is_context:
#         # NOTE: stripping accent requires adjusting answer span accordingly,
#         # not stripping accent will incur more unknown tokens,
#         # and can happen in answer, here we strip a word only if its length
#         # is the same as being stripped accents
#         text = ' '.join([strip_accent(c)
#                          if len(c) == len(strip_accent(c)) else c
#                          for c in input_text.split(' ')])
#     else:
#         text = strip_accent(input_text)
#     text_tokens = word_tokenize(text)
#     text_wp_tokens, text_ids = text_to_word_piece(text_tokens, vocab=vocab)
#     return text, text_ids, text_wp_tokens


# BERT_VOCAB = None
#
#
# def text_to_word_piece(tokens, vocab=None, unk_token='[UNK]',
#                        char_len_limit=100):
#     if vocab is None:
#         # use bert vocab
#         global BERT_VOCAB
#         if BERT_VOCAB is None:
#             BERT_VOCAB = load_vocab()
#         vocab = BERT_VOCAB
#     output_tokens = []
#     output_ids = []
#     for token in tokens:
#         chars = list(token)
#         if len(chars) > char_len_limit:
#             output_tokens.append(token)
#             output_ids.append(vocab[unk_token])
#             continue
#         is_bad = False
#         start = 0
#         sub_tokens = []
#         sub_ids = []
#         while start < len(chars):
#             end = len(chars)
#             cur_substr = None
#             while start < end:
#                 substr = "".join(chars[start:end])
#                 if start > 0:
#                     substr = "##" + substr
#                 if substr in vocab:
#                     cur_substr = substr
#                     break
#                 end -= 1
#             if cur_substr is None:
#                 is_bad = True
#                 break
#             sub_tokens.append(cur_substr)
#             sub_ids.append(vocab[cur_substr])
#             start = end
#         if is_bad:
#             output_tokens.append(token)
#             output_ids.append(vocab[unk_token])
#         else:
#             output_tokens.extend(sub_tokens)
#             output_ids.extend(sub_ids)
#
#     return output_tokens, output_ids


# def strip_accent(text):
#     # 'régime' ==> 'regime', 'François' ==> 'Francois'
#     # tested samples at https://en.wikipedia.org/wiki/Diacritic
#     # 'Ç, Ğ, I, İ, Ö, Ş' ==> 'C, G, I, I, O, S'
#     # 'à, ç, é, è, í, ï, ó, ò, ú, ü' ==> 'a, c, e, e, i, i, o, o, u, u'
#     return ''.join(c for c in unicodedata.normalize('NFD', text)
#                    if unicodedata.category(c) != 'Mn')


# def should_keep(token):
#     """filter english stopwords and punctuation"""
#     token = normalize(token)
#     return not (re.match(r'^\p{P}+$', token) or token.lower() in STOP_WORDS)


# def filter_words(words):
#     return list(filter(should_keep, words))

def get_valid_windows(num_tokens, max_num_tokens, window_stride):
    # use sliding window to take tokens within max_context_tokens limit, e.g.:
    # get_valid_windows(300, 317, 128) ==> [(0, 300)]
    # get_valid_windows(400, 317, 128) ==> [(0, 317), (128, 400)]
    # get_valid_windows(700, 317, 128) ==> [(0, 317), (128, 445), (256, 573),
    # (384, 700)]
    window_spans = []
    start_offset = 0
    if window_stride is None or window_stride < 0:
        window_stride = 0
    while start_offset < num_tokens:
        length = num_tokens - start_offset
        if length > max_num_tokens:
            length = max_num_tokens
        window_spans.append((start_offset, start_offset + length))
        if start_offset + length == num_tokens or window_stride == 0:
            break
        start_offset += min(length, window_stride)
    return window_spans


# def get_answer_token_span(token_spans, answer_char_span, debug_info=None):
#     answer_char_start, answer_char_end = answer_char_span
#     answer_token_start, answer_token_end = None, None
#
#     for token_idx, token_span in enumerate(token_spans):
#         token_char_start, token_char_end = token_span
#         if token_char_start <= answer_char_start <= token_char_end:
#             answer_token_start = token_idx
#         if token_char_start <= answer_char_end <= token_char_end:
#             answer_token_end = token_idx
#         if answer_token_start and answer_token_end:
#             break
#     if answer_token_start is None or answer_token_end is None:
#         if debug_info:
#             context, tokens = debug_info
#             token_str = ['idx={}, token={}, span={}'.format(i, t, s)
#                          for i, (t, s) in enumerate(zip(tokens, token_spans))]
#             debug_str = ['no answer_char_span: {},'.format(answer_char_span),
#                          'tokens: {},'.format('\n\t'.join(token_str)),
#                          'context={}'.format(context)]
#             raise ValueError('\n'.join(debug_str))
#         else:
#             error_str = ['no answer_char_span: {},'.format(answer_char_span),
#                          ' spans: {}\n'.format(token_spans)]
#             raise ValueError('\n'.join(error_str))
#     return answer_token_start, answer_token_end


def decode_answer(contexts, context_spans, start_predictions, end_predictions,
                  output_char_start=False):
    answers = []
    for ctx, ctx_span, start_pos, end_pos in zip(contexts, context_spans,
                                                 start_predictions,
                                                 end_predictions):
        start_span = ctx_span[min(start_pos, len(ctx_span) - 1)]
        predicted_char_start = start_span[0]
        end_span = ctx_span[min(end_pos, len(ctx_span) - 1)]
        predicted_char_end = end_span[1]
        predicted_text = ctx[predicted_char_start:predicted_char_end]
        if output_char_start:
            answers.append((predicted_char_start, predicted_text))
        else:
            answers.append(predicted_text)
    return answers


def token_to_index(tokens, vocab):
    return [vocab.get(t, 1) for t in tokens]
