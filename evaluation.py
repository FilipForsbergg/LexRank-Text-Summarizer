import numpy as np
from LexRank.LexRank import LexRank, CNNDailyMailCorpus
from rouge_score import rouge_scorer


def evaluate_lexrank_vs_lead_for_n(
    n_articles: int,
    n_sentences: int,
    corpus_size: int = 1000,
):
    """
    Kör utvärdering av LexRank och Lead-k mot CNN/DailyMail-highlights
    för ett givet antal meningar (n_sentences).

    Returnerar en dict med mean och std för varje modell och ROUGE-mått.
    """

    print(f"\n=== Evaluating n_sentences = {n_sentences} ===")
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
        reference = sample.highlights  # referens-sammanfattning

        if not article or not reference:
            continue

        # LexRank-sammanfattning
        lex_sentences = lexrank.summarize(article, n_sentences=n_sentences)
        lex_summary = " ".join(lex_sentences)

        # Lead-k-baseline
        lead_sentences = lexrank.lead_k_base_line(article, k=n_sentences)
        lead_summary = " ".join(lead_sentences)

        # ROUGE
        lex_rouge = scorer.score(reference, lex_summary)
        lead_rouge = scorer.score(reference, lead_summary)

        for key in lex_scores.keys():
            lex_scores[key].append(lex_rouge[key].fmeasure)
            lead_scores[key].append(lead_rouge[key].fmeasure)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_articles} articles...")

    def stats(d):
        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
            for k, v in d.items()
            if len(v) > 0
        }

    lex_stats = stats(lex_scores)
    lead_stats = stats(lead_scores)

    print("\n===== AVERAGE ROUGE (F1) =====")
    print(f"On {n_articles} articles, {n_sentences} sentences per summary\n")

    print("LexRank:")
    for k, v in lex_stats.items():
        print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    print("\nLead-k (first sentences):")
    for k, v in lead_stats.items():
        print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    # Returnera strukturerat så vi kan bygga tabell
    return {
        "n_sentences": n_sentences,
        "lexrank": lex_stats,
        "lead": lead_stats,
    }


def sweep_n_values(
    n_values=(1, 2, 3, 4, 5),
    n_articles: int = 100,
    corpus_size: int = 1000,
):
    """
    Kör utvärdering för flera olika n (antal meningar)
    och skriver ut en tabell som kan användas i rapporten.
    """

    results = []

    for n in n_values:
        res = evaluate_lexrank_vs_lead_for_n(
            n_articles=n_articles,
            n_sentences=n,
            corpus_size=corpus_size,
        )
        results.append(res)

    # Bygg en enkel tabell (Markdown-stil) som du kan klistra in i rapporten
    print("\n\n================ TABLE (Markdown-style) ================\n")
    print("| n | Model   | ROUGE | Mean  | Std   |")
    print("|---|---------|-------|-------|-------|")

    for res in results:
        n = res["n_sentences"]
        for model_name in ["lexrank", "lead"]:
            model_label = "LexRank" if model_name == "lexrank" else "Lead-k"
            for rouge_key in ["rouge1", "rouge2", "rougeL"]:
                m = res[model_name][rouge_key]["mean"]
                s = res[model_name][rouge_key]["std"]
                print(f"| {n} | {model_label} | {rouge_key} | {m:.4f} | {s:.4f} |")

    return results


if __name__ == "__main__":
    # Exempel: kör för n = 1..5
    sweep_n_values(
        n_values=(1, 2, 4, 6, 8,),
        n_articles=100,
        corpus_size=1000,
    )
