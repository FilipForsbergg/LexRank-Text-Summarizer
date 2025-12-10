import numpy as np
from LexRank.LexRank import LexRank, CNNDailyMailCorpus
from rouge_score import rouge_scorer

def evaluate_lexrank_vs_lead(
    n_articles: int = 100,
    n_sentences: int = 3,
    corpus_size: int = 1000,
):
    """
    Kör utvärdering av LexRank och Lead-k mot CNN/DailyMail-highlights.
    n_articles <= corpus_size.
    """

    print(f"Loading CNN/DailyMail subset (size={corpus_size})...")
    corpus = CNNDailyMailCorpus(amount=corpus_size)

    n_articles = min(n_articles, len(corpus))
    print(f"Evaluating on {n_articles} articles.\n")

    lexrank = LexRank()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    lex_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    lead_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i in range(n_articles):
        sample = corpus[i]

        article = sample.article
        reference = sample.highlights #amek sure its a string

        if not article or not reference:
            continue

        lex_sentences = lexrank.summarize(article, n_sentences=n_sentences)
        lex_summary = " ".join(lex_sentences)

        lead_sentences = lexrank.lead_k_base_line(article, k=n_sentences)
        lead_summary = " ".join(lead_sentences)

        lex_rouge = scorer.score(reference, lex_summary)
        lead_rouge = scorer.score(reference, lead_summary)

        for key in lex_scores.keys():
            lex_scores[key].append(lex_rouge[key].fmeasure)
            lead_scores[key].append(lead_rouge[key].fmeasure)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_articles} articles...")

    def avg(d):
        return {k: float(np.mean(v)) for k, v in d.items() if len(v) > 0}

    lex_avg = avg(lex_scores)
    lead_avg = avg(lead_scores)

    print("\n===== AVERAGE ROUGE (F1) =====")
    print(f"On {n_articles} articles, {n_sentences} sentences per summary\n")

    print("LexRank:")
    for k, v in lex_avg.items():
        print(f"  {k}: {v:.4f}")

    print("\nLead-k (first sentences):")
    for k, v in lead_avg.items():
        print(f"  {k}: {v:.4f}")

    return lex_avg, lead_avg

if __name__ == "__main__":
    evaluate_lexrank_vs_lead(
        n_articles=100,
        n_sentences=3,
        corpus_size=1000,
    )