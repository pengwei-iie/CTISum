import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
from collections import deque

import numpy as np
import pandas as pd
import torch
from multiprocess import Pool

# from log import init_logger
# import data_builder

from log import init_logger
from log import logger
from tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from utils import clean
from prepro_utils import _get_word_ngrams
import multiprocessing

import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)
    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])

    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])


    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    return sorted(selected)



def format_attack_to_data_and_index(args):
    for type in ['valid', 'test', 'train']:
        logger.info('Processing %s' % pjoin(args.raw_path + '/' + type))

        b_lst = [(pjoin(args.raw_path + '/' + type, f), args) for f in os.listdir(pjoin(args.raw_path, type))]
        pool = Pool(args.n_cpus)
        dataset = []
        for d in pool.imap_unordered(_format_attack_to_data_and_index, b_lst):
            dataset.append(d)

        data_list = []
        index_list = []
        for i, d in enumerate(dataset):
            source, tgt = d['src'], d['tgt']
            text = [" ".join(sent) for sent in source]
            summary = [" ".join(sent) for sent in tgt]

            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 5)
            data_dict = {
                'text': text,
                'summary': summary
            }
            data_list.append(data_dict)

            index_dict = {
                'sent_id': sent_labels
            }
            index_list.append(index_dict)

        p_ct = 0
        data_file = "{:s}/{:s}/{:s}_{:d}.jsonl".format(args.save_path, 'data', type, p_ct)
        index_file = "{:s}/{:s}/{:s}_{:d}.jsonl".format(args.save_path, 'index', type, p_ct)
        with open(data_file, 'w') as jsonl_file:
            for data_dict in data_list:
                jsonl_file.write(json.dumps(data_dict) + '\n')
        with open(index_file, 'w') as jsonl_file:
            for index_dict in index_list:
                jsonl_file.write(json.dumps(index_dict) + '\n')

        logger.info('Processed instances %d' % len(dataset))
        logger.info('Saving to %s' % data_file)
        logger.info('Saving to %s' % index_file)


def _format_attack_to_data_and_index(params):
    f, args = params
    source, target = load_json(f, args.lower)

    target = [' '.join(sent) for sent in target]
    target = re.split(r'(\d+ \. )', target[0])
    target = [item for item in target if item]

    tar = []
    for i, m in enumerate(target):
        if i % 2 == 1:
            tar.append(m)

    tgt = []
    for sent in tar:
        item = sent.split(' ')
        tgt.append(item)
    return {'src': source, 'tgt': tgt}

def format_general_to_data_and_index(args):
    for type in ['valid', 'test', 'train']:
        logger.info('Processing %s' % pjoin(args.raw_path + '/' + type))

        b_lst = [(pjoin(args.raw_path + '/' + type, f), args) for f in os.listdir(pjoin(args.raw_path, type))]
        pool = Pool(args.n_cpus)
        dataset = []
        for d in pool.imap_unordered(_format_general_to_data_and_index, b_lst):
            dataset.append(d)

        data_list = []
        index_list = []
        for i, d in enumerate(dataset):
            source, tgt = d['src'], d['tgt']
            text = [" ".join(sent) for sent in source]
            summary = [" ".join(sent) for sent in tgt]

            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 5)
            data_dict = {
                'text': text,
                'summary': summary
            }
            data_list.append(data_dict)

            index_dict = {
                'sent_id': sent_labels
            }
            index_list.append(index_dict)

        p_ct = 0
        data_file = "{:s}/{:s}/{:s}_{:d}.jsonl".format(args.save_path, 'data', type, p_ct)
        index_file = "{:s}/{:s}/{:s}_{:d}.jsonl".format(args.save_path, 'index', type, p_ct)
        with open(data_file, 'w') as jsonl_file:
            for data_dict in data_list:
                jsonl_file.write(json.dumps(data_dict) + '\n')
        with open(index_file, 'w') as jsonl_file:
            for index_dict in index_list:
                jsonl_file.write(json.dumps(index_dict) + '\n')

        logger.info('Processed instances %d' % len(dataset))
        logger.info('Saving to %s' % data_file)
        logger.info('Saving to %s' % index_file)


def _format_general_to_data_and_index(params):
    f, args = params
    source, target = load_json(f, args.lower)

    target = [' '.join(sent) for sent in target]
    target = target[0].split('.')
    target = [item for item in target if item]

    tgt = []
    for sent in target:
        item = sent.split(' ')
        item = [word for word in item if word]
        tgt.append(item)

    return {'src': source, 'tgt': tgt}


