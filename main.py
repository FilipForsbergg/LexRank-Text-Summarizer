from LexRank.LexRank import LexRank, CNNDailyMailCorpus

def main(articles):
    corpus = CNNDailyMailCorpus(articles)
    lexrank = LexRank()

    sample = corpus[0]
    article = sample.article if len(sample.article) < 10000 else sample.article[:1000] + "...\n"
    summary_sentences = lexrank.summarize(sample.article, n_sentences=2)
    summary = " ".join(summary_sentences)

    
    print("ARTICLE: \n", article, "\n")
    print("LexRank summary: \n", summary, "\n")
    print("REFERENCE HIGHLIGHTS: \n", sample.highlights)

if __name__ == "__main__":
    articles = "data/cnn_500.jsonl"
    main(articles)