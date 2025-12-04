import numpy as np
from numpy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from dataclasses import dataclass
from typing import List, Iterator
from nltk.corpus import stopwords
import re
from datasets import load_dataset

class LexRank:
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

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )
    
    #public methods
    def summarize(
            self,
            text: str, 
            n_sentences: int
        ) -> list[str]:

        sentences = self._split_sentences(text)
        if len(sentences) == 0:
            return []
        if n_sentences >= len(sentences):
            return sentences
        
        tfidf = self._build_tfidf(sentences)
        sim = self._build_similarity_matrix(tfidf)
        adj = self._build_adjacency_matrix(sim)
        scores = self._pagerank(adj)

        # we pcik top n sentences as the summary
        ranked_indices  =np.argsort(-scores) # descending
        selected = sorted(ranked_indices[:n_sentences])
        summary = [sentences[i] for i in selected]
        return summary

    def lead_k_base_line(self, text: str, k: int) -> list[str]:
        """
        Returns the first k sentences of the text.
        Used as a baseline.
        """
        return self._split_sentences(text)[:k]

    #helper methods
    def _split_sentences(self, text: str) -> list[str]:
        #simple tokenizer
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def _build_tfidf(self, sentences):
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        return tfidf_matrix
    
    def _build_adjacency_matrix(self, sim_matrix):
        adj = np.where(sim_matrix > self.threshold, sim_matrix, 0.0)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        adj = adj / row_sums
        return adj
    
    def _build_similarity_matrix(self, tfidf_matrix):
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0.0) # so sentences don't link to themselves
        return sim_matrix

    def _pagerank(self, adj_matrix):
        n = adj_matrix.shape[0]
        pr = np.ones(n) / n

        for _ in range(self.max_iter):
            prev_pr = pr.copy()
            pr = (
                (1 - self.damping) / n
                + self.damping * np.dot(adj_matrix.T, pr)
            )
            if linalg.norm(pr - prev_pr, ord=1) < self.tol:
                break
        return pr

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
        text_to_sum = random_article["article"]
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
                    highlights=["highlights"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ArticleSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[ArticleSample]:
        return iter(self.samples)