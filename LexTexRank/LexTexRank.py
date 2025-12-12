from __future__ import annotations

import numpy as np
from numpy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
from typing import List
from dataclasses import dataclass
from typing import List, Iterator
from datasets import load_dataset

class BaseGraphRankSummarizer:
    """
    Bas-klass för LexRank och TextRank.
    Implementerar:
      - meningsdelning
      - adjacency-matris (med threshold)
      - PageRank
      - summarize()-flödet
    _build_similarity_matrix(sentences) defineras i respektive subklass.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        tol: float = 1e-6,
        damping: float = 0.85,
        max_iter: int = 100,
    ):
        self.threshold = threshold
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter

    #public
    def summarize(self, text: str, n_sentences: int) -> List[str]:
        """
        Main method that returns the best n sentences.
        """
        sentences = self._split_sentences(text)

        if len(sentences) == 0:
            return []
        if n_sentences >= len(sentences):
            return sentences

        #1. Similarity matrix (dependent on the subclass)
        sim = self._build_similarity_matrix(sentences)

        #2. Adjacency matrix
        adj = self._build_adjacency_matrix(sim)

        #3. PageRank on the adjacency matrix
        scores = self._pagerank(adj)

        ranked_indices = np.argsort(-scores)
        selected = sorted(ranked_indices[:n_sentences])

        return [sentences[i] for i in selected]

    def lead_k_base_line(self, text: str, k: int) -> List[str]:
        return self._split_sentences(text)[:k]

    #helper methods
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        #the sub lcass must implement this
        raise NotImplementedError

    def _split_sentences(self, text: str) -> List[str]:
        """Enkel meningssegmentering med regex."""
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _build_adjacency_matrix(self, sim_matrix: np.ndarray) -> np.ndarray:
        adj = np.where(sim_matrix > self.threshold, sim_matrix, 0.0)

        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        adj = adj / row_sums
        return adj

    def _pagerank(self, adj_matrix: np.ndarray) -> np.ndarray:
        n = adj_matrix.shape[0]
        pr = np.ones(n) / n  #start uniform

        for _ in range(self.max_iter):
            prev_pr = pr.copy()
            pr = (
                (1 - self.damping) / n
                + self.damping * np.dot(adj_matrix.T, pr)
            )
            if linalg.norm(pr - prev_pr, ord=1) < self.tol:
                break
        return pr

class LexRank(BaseGraphRankSummarizer):
    """
    LexRank: uses TF-IDF and cosine-similarity
    """

    def __init__(
        self,
        threshold: float = 0.1,
        tol: float = 1e-6,
        damping: float = 0.85,
        max_iter: int = 100,
        language: str = "english",
    ):
        super().__init__(threshold=threshold, tol=tol, damping=damping, max_iter=max_iter)

        self.language = language
        stop_words = self._get_stopwords(language)

        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            lowercase=True,
        )

    def _get_stopwords(self, language: str = "english") -> List[str] | None:
        """Returns stopwords for given language"""
        try:
            return stopwords.words(language)
        except LookupError:
            nltk.download("stopwords", quiet=True)
            try:
                return stopwords.words(language)
            except LookupError:
                return None

    #based on cosine similarity
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0.0)
        return sim_matrix


class TextRank(BaseGraphRankSummarizer):
    """
    TextRank: använder ordöverlapp som likhetsmått mellan meningar.
    """

    def __init__(
        self,
        threshold: float = 0.0,  # usually no hard threshold with TextRank
        tol: float = 1e-6,
        damping: float = 0.85,
        max_iter: int = 100,
    ):
        super().__init__(threshold=threshold, tol=tol, damping=damping, max_iter=max_iter)

    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """likhetsmatris baserat på ordöverlapp"""
        n = len(sentences)
        sim = np.zeros((n, n))

        tokenized = [
            [w for w in re.findall(r"\w+", s.lower()) if w]
            for s in sentences
        ]

        for i in range(n):
            words_i = set(tokenized[i])
            len_i = len(words_i)
            if len_i == 0:
                continue

            for j in range(n):
                if i == j:
                    continue
                words_j = set(tokenized[j])
                len_j = len(words_j)
                if len_j == 0:
                    continue

                common = len(words_i & words_j)
                if common == 0:
                    continue

                # Klassisk TextRank-likhet:
                # common / (log(|Si|) + log(|Sj|))
                denom = np.log(len_i) + np.log(len_j)
                if denom <= 0:
                    continue
                sim[i, j] = common / denom

        np.fill_diagonal(sim, 0.0)
        return sim
    
@dataclass
class ArticleSample:
    id: str
    article: str
    highlights: str

class CNNDailyMailCorpus:
    def __init__(self, amount: int = 5000):
        self.samples: List[ArticleSample] = []
        self.n = amount    
        self._load()
    
    #public
    def load_random_article(self):
        """
        Picks a random CNN article
        """
        rx: int = np.random.choice(self.n)
        random_article = self.samples[rx]
        text_to_sum = random_article.article
        return text_to_sum
    
    def _load(self):
        """
        Loads a subset of CNN/DailyMail articles
        """
        split="train"
        n_samples=self.n
        
        subset_spec = f"{split}[:{n_samples}]"
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=subset_spec)

        for a in dataset:
            self.samples.append(
                ArticleSample(
                    id=a["id"],
                    article=a["article"],
                    highlights=a["highlights"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ArticleSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[ArticleSample]:
        return iter(self.samples)