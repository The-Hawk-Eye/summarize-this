"""
Utility functions for calculating summary evaluation metrics.
"""

import os
import numpy as np
from nltk.tokenize import TweetTokenizer
from string import punctuation
from typing import List, Tuple

from text_summarizer import Summarizer
from keyword_extractor import Extractor

tokenizer = TweetTokenizer()


def count_ngrams(model_summary: str, gold_summary: str, n: int=2) -> Tuple[int, int, int]:
    """
    Given a model-generated summary and a gold summary, counts the n-grams in both summaries, and
    counts the number of intersecting n-grams.

    @param model_summary (str): raw text, model-generated summary.
    @param gold_summary (str): raw text, gold summary.
    @param n (int): n-gram length.
    @returns ms_ngrams, gs_ngrams, intersect_ngrams (Tuple[int, int, int]): a tuple of ints giving:
            - the number of n-grams in the model summary
            - the number of n-grams in the gold summary
            - the number of intersecting n-grams between both summaries
    """
    summary_tokens = tokenizer.tokenize(model_summary)
    summary_tokens = [tok_.lower() for tok_ in summary_tokens if tok_ not in punctuation]
    summary_ngrams = []
    for i in range(len(summary_tokens) - n + 1):
        summary_ngrams.append(summary_tokens[i : n + i])

    gold_summary_tokens = tokenizer.tokenize(gold_summary)
    gold_summary_tokens = [tok_.lower() for tok_ in gold_summary_tokens if tok_ not in punctuation]
    gold_summary_ngrams = []
    for i in range(len(gold_summary_tokens) - n + 1):
        gold_summary_ngrams.append(gold_summary_tokens[i : n + i])

    intersect_ngrams = []
    for ngram in summary_ngrams:
        if ngram in gold_summary_ngrams:        
            intersect_ngrams.append(ngram)

    return (len(summary_ngrams), len(gold_summary_ngrams), len(intersect_ngrams))


def ROUGE_N(model_summary: str, gold_summary: str, n: int=2) -> float:
    """
    Given a model summary and a gold summary calculates the ROUGE-N score of the model summary.

    @param model_summary (str): raw text, model-generated summary.
    @param gold_summary (str): raw text, gold summary.
    @param n (int): n-gram length.
    @returns rouge_score (float): ROUGE-N score of the model summary.
    """
    _, gold_ngrams, intersect_ngrams = count_ngrams(model_summary, gold_summary, n)
    if gold_ngrams:
        rouge_score = intersect_ngrams / gold_ngrams
    else:
        rouge_score = 0

    return rouge_score



def BLEU(model_summary: str, gold_summary: str, n: int=4) -> float:
    """
    Given a model summary and a gold summary calculates the BLEU score of the model summary.

    @param model_summary (str): raw text, model-generated summary.
    @param gold_summary (str): raw text, gold summary.
    @param n (int): maximum n-gram length to count number of intersecting n-grams for.
    @returns bleu_score (float): BLEU score of the model summary.
    """
    bleu_score = 0.

    for i in range(n):
        summary_ngrams, _, intersect_ngrams = count_ngrams(model_summary, gold_summary, i)
        if summary_ngrams and intersect_ngrams:
            bleu_score += np.log2(intersect_ngrams / summary_ngrams)

    bleu_score /= n
    bleu_score = 2 ** bleu_score

    return bleu_score


def summarizer_evaluation(summarizer: Summarizer, data_path: str, rougeN: int=2, maxN: int=4) -> Tuple[float, float]:
    """
    Evaluates a model Summarizer on the BBC News Summary corpus of data. Calculates the corpus-level
    rouge-n score and the corpus-level bleu score.

    @param summarizer (Summarizer): Summarizer object.
    @param data_path (str): file path to the corpus data.
    @param rougeN (int): length of n-grams to calculate ROUGE-N score on.
    @param maxN (int): maximum n-gram length to calculate BLEU score on.
    @returns rouge_score, bleu_score (Tuple[float, float]): corpus-level rouge-n score and corpus-level
                                                            bleu score.
    """
    text_path = data_path + "/News_Articles/"
    summaries_path = data_path + "/Summaries/"

    news_topics = []
    for dir_ in os.scandir(text_path):
        if dir_.is_dir():
            news_topics.append(dir_.name)

    rouge_score = bleu_score = 0.
    num_topics = 0
    for topic_ in news_topics:
        num_topics += 1
        num_docs = 0
        topic_rouge_score = topic_bleu_score = 0.

        for doc_ in os.scandir(text_path + topic_):
            if  not doc_.is_file():
                continue
            num_docs += 1
            gold_summary = open(summaries_path + topic_ + "/" + doc_.name).read()
            model_summary = summarizer.summarize(open(doc_).read())
            topic_rouge_score += ROUGE_N(model_summary, gold_summary, n=rougeN)
            topic_bleu_score += BLEU(model_summary, gold_summary, n=maxN)

        rouge_score += topic_rouge_score / num_docs
        bleu_score += topic_bleu_score / num_docs

    rouge_score /= num_topics
    bleu_score /= num_topics

    return rouge_score, bleu_score