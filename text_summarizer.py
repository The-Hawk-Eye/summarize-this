import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from typing import List

tokenizer = TweetTokenizer()
nltk.download("stopwords")
stop_words = stopwords.words("english")


"""
Summarizer object. Used to create text summary.
The model used Google Page rank algorithm to rank sentences in the text.
Afterwards the model selects the top scored sentences to create text summary.
"""
class Summarizer:
    def __init__(self, ratio: int=0.4, damp: int=0.85, epsi: int=0.0001):
        """
        Init summarizer object.

        @param ratio (float): relative length of the summary to be created. [default: 0.4]
        @param damp (float): damping factor for calculating the stationary distribution. [default: 0.85]
        @param epsi (float): treshold between the theoretical stationary distribution and the
                             numerically calculated stationary distribution. [default: 0.0001]
        """
        self.ratio = ratio
        self.damp = damp
        self.epsi = epsi
        self.Q = None       # transition matrix
        self.N = 0          # number of sentences in the document


    def text_segmentation(self, content: str) -> List[str]:
        """
        This function splits the given text into a list of sentences.

        @param content (str): raw text.
        @returns sents (List[str]): a list of sentences.
        """
        return sent_tokenize(content)


    def sentence_tokenization(self, sentence: str) -> List[str]:
        """
        This function splits a sentence into a list of words.

        @param sentence (str): sentence
        @returns words (List[str]): a list of words.
        """
        sent_tokens = tokenizer.tokenize(sentence)
        return [tok_.lower() for tok_ in sent_tokens if tok_ not in punctuation and tok_ not in stop_words]


    def _overlap_similarity(self, S1: List[str], S2: List[str]) -> float:
        """
        Calculates the similarity between two sentences using overlapping.

        @param S1 (List[str]): sentence, a list of words.
        @param S2 (List[str]): sentence, a list of words.
        @returns score (float): similarity score between S1 and S2.
        """
        if len(S1) == 0 or len(S2) == 0:
            return 0

        words_in_common = set()
        sent_2 = set(S2)
        for tok_ in S1:
            if tok_ in sent_2:
                words_in_common.add(tok_)
        score = len(words_in_common)

        # normalization to avoid promoting long sentences
        score /= (np.log2(len(S1) + 1) + np.log2(len(S2) + 1))

        return score


    def _build_transition_matrix(self, tokenized_sentences: List[List[int]]) -> None:
        """
        Given a similarity score function builds the similarity matrix.

        @param tokenized_sentences (List[List[str]]): list of sentences, each sentence is a list of words.
        """
        self.N = len(tokenized_sentences)
        if not self.N:
            raise ValueError("No sentences to summarize!")

        self.Q = np.zeros([self.N, self.N])

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                self.Q[i, j] = self._overlap_similarity(tokenized_sentences[i], tokenized_sentences[j])

            if self.Q[i].sum():
                self.Q[i] /= self.Q[i].sum()    # normalizing each row of the matrix

        # Google PageRank modification
        self.Q = self.damp * self.Q + (1 - self.damp) / self.N


    def _find_scores(self):
        """
        This function calculates the stationary distribution of the similarity matrix.

        @returns scores (vector): Numpy array of length (self.N) representing the stationary distribution.
        """
        # Initial distribution p.
        scores = np.array([1 / self.N] * self.N)

        # At each step compute pQ. If p == pQ then p is the stationary distribution.
        while True:
            new_scores = scores.dot(self.Q)
            delta = abs(new_scores - scores).sum()
            if delta <= self.epsi:
                break
            scores = new_scores

        return scores


    def summarize(self, content: str) -> str:
        """
        Summarizes the text.

        @param content (str): raw text.
        @returns summary(str): summary of the text.
        """
        # Split the text into a list of sentences.
        sentences = self.text_segmentation(content)

        # Split each sentence into a list of words.
        tokenized_sentences = [self.sentence_tokenization(sent) for sent in sentences]

        # Build the transition matrix and compute the stationary distribution.
        self._build_transition_matrix(tokenized_sentences)
        scores = self._find_scores()

        # Get the "top_k" scored sentences. Sort the sentences by score and retrieve the indecies
        # of the "top_k" sentences. After that sort the indecies to arrange the sentences in order
        # of appearance in the text.
        top_k = int(self.ratio * self.N)
        top_scores = np.argsort(scores)[::-1][:top_k]
        top_scores = np.sort(top_scores)

        # Build the summary by concatenating the sentences with the top scores.
        summary = " ".join([sentences[idx] for idx in top_scores])

        return summary