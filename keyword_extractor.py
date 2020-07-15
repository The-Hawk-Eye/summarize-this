import numpy as np
import nltk
import spacy
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from typing import List

tokenizer = TweetTokenizer()
nltk.download("stopwords")
stop_words = stopwords.words("english")
nlp = spacy.load("en_core_web_sm")


"""
Extractor object. Used to extract key words from text.
The model used Google Page rank algorithm to rank the words in the text.
Afterwards the model selects the top scored words as key words.
"""
class Extractor:
    def __init__(self, window_size: int=4, num_words: int=20, damp: float=0.85, epsi: float=0.0001):
        """
        Init extractor object.

        @param window_size (int): window size to consider co-occuring of words. [default: 4]
        @param num_words (int): number of key words to be extracted. [default: 20]
        @param damp (float): damping factor for calculating the stationary distribution. [default: 0.85]
        @param epsi (float): treshold between the theoretical stationary distribution and the
                             numerically calculated stationary distribution. [default: 0.0001]
        """
        self.window_size = window_size
        self.num_words = num_words
        self.damp = damp
        self.epsi = epsi
        self.word2id = {}
        self.id2word = {}
        self.Q = None                   # transition matrix
        self.N = 0                      # number of candidate words


    def POS_tagging(self, content: str, candidate_pos: List[str]) -> List[List[str]]:
        """
        Given text conctent and a list of part-of-speech tags, for every sentence selects only words
        of a certain part of speech.

        @param content (str): raw text.
        @param candidate_pos (List[str]): list of part-of-speech tags.
        @returns pos_tagged_sentences (List[List[str]]): a list of sentences, every sentence is a list of words.
                                                         Keep only words that have relevant pos tags.
        """
        doc = nlp(content)
        pos_tagged_sentences = [[tok_.text for tok_ in sent if tok_.pos_ in candidate_pos and tok_.is_stop is False]
                                           for sent in doc.sents]
        return pos_tagged_sentences


    def _build_transition_matrix(self, pos_tagged_sentences: List[List[str]]) -> None:
        """
        Given a list of sentences, builds a dictionary of candidate words. The candidate words are vertices of
        the graph. For every sentence scans a window of size wondow_size. For every pair of words in every window
        adds an edge in the graph. Builds a transition matrix over the graph.

        @param pos_tagged_sentences (List[List[str]]): a list of sentences, every sentence is a list of words.
        """
        for sent in pos_tagged_sentences:
            for word in sent:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        self.id2word = {v : k for k, v in self.word2id.items()}

        self.N = len(self.word2id)
        if not self.N:
            raise ValueError("No candidate words in the vocabulary!")

        self.Q = np.zeros([self.N, self.N])

        word_pairs = []
        for sent in pos_tagged_sentences:
            for i in range(len(sent) - 1):
                window_size = min(self.window_size, len(sent) - i)
                for j in range(i + 1, i + window_size):
                    word1, word2 = sent[i], sent[j]
                    ind1, ind2 = self.word2id[word1], self.word2id[word2]
                    self.Q[ind1, ind2] = self.Q[ind2, ind1] = 1

        # normalize matrix rows
        for i in range(self.N):
            if self.Q[i].sum():
                self.Q[i] /= self.Q[i].sum()

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


    def extract(self, content: str, candidate_pos: List[str]) -> List[str]:
        """
        Extract keywords from text.

        @params content (str): raw text.
        @param candidate_pos (List[str]): list of part-of-speech tags.
        @returns keywords (List[str]): a list of keywords.
        """
        # Split the text into a list of list of words. Keep only words that have a relevant pos tag.
        pos_tagged_sentences = self.POS_tagging(content, candidate_pos)

        # Build the transition matrix and compute the stationary distribution.
        self._build_transition_matrix(pos_tagged_sentences)
        scores = self._find_scores()

        # Get the "top_k" scored words. Sort the words by score and retrieve the indecies
        # of the "top_k" words. After that look-up the indecies to retrueve the words.
        top_k = min(self.num_words, len(self.word2id) // 3) # at least 1/3 of the words are extracted
        word_ids = np.argsort(scores)[::-1][:top_k]
        keywords = [self.id2word[idx] for idx in word_ids]

        return keywords