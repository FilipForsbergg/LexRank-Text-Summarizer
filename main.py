import numpy as np
from numpy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

class LexRank:
    def __init__(
            self,
            threshold: float = 0.1,
            tol: float = 1e-6,
            damping: float = 0.85,
            language: str = "swedish",
            max_iter: int = 100,
    ):
        self.threshold = threshold
        self.damping = damping
        self.tol = tol
        self.language = language
        self.max_iter = max_iter


        swedish_stopwords = self._get_swedish_stopwords() #och, det, att etc...
        self.vectorizer = TfidfVectorizer(
            stop_words=swedish_stopwords,
            lowercase=True,
        )
    
    #public methods
    def summarize(
            self,
            text: str, 
            n_sentences: int
        ) -> list[str | None]:

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

        return [sentences[i] for i in selected]


    #helper methods
    def _get_swedish_stopwords(self) -> list[str]:
        try:
            return stopwords.words('swedish')
        except LookupError:
            nltk.download('stopwords')
            return stopwords.words('swedish')

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
    
# Example test text to summarize
if __name__ == "__main__":
    text = """
    Klimatförändringar påverkar världen på många sätt. Extrema väderhändelser som värmeböljor, torka och översvämningar blir allt vanligare.
    Forskare varnar för att den globala uppvärmningen kan leda till stora förändringar i ekosystem och livsmedelsförsörjning.
    För att begränsa temperaturökningen krävs kraftiga minskningar av utsläppen av växthusgaser.
    Många länder investerar nu i förnybar energi, som sol- och vindkraft.
    Samtidigt behövs internationellt samarbete för att hantera de långsiktiga effekterna av klimatförändringarna.
    """

    lr = LexRank(threshold=0.1)
    summary = lr.summarize(text, n_sentences=2)
    print("------------------------------")
    print("           Summering          ")
    print("------------------------------")
    for s in summary:
        print("-", s)