from LexTexRank.LexTexRank import LexRank, TextRank, CNNDailyMailCorpus
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='LexRank / TextRank text summarization')
    parser.add_argument('--file', '-f', type=str, help='Path to text file to summarize')
    parser.add_argument('--n_sentences', '-n', type=int, default=3, help='Number of sentences in the summary')
    parser.add_argument('--model', '-m', type=str, required=True, choices=["lexrank", "textrank", "leadk"], help='Summarization method: ["lexrank", "textrank", "leadk"]')

    args = parser.parse_args()

    lexrank = LexRank()
    textrank = TextRank()

    original_text = None
    summary = None
    n = args.n_sentences

    #Summarize from file
    if args.file:
        file_path = args.file.strip()
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        if args.model == "lexrank":
            summary = lexrank.summarize(text=text_content, n_sentences=n)
        elif args.model == "textrank":
            summary = textrank.summarize(text=text_content, n_sentences=n)
        else:
            summary = lexrank.lead_k_base_line(text=text_content, k=n)

    #pick a random article
    else:
        cnn = CNNDailyMailCorpus()
        original_text = cnn.load_random_article()
        if args.model == "lexrank":
            summary = lexrank.summarize(text=original_text, n_sentences=n)
        elif args.model == "textrank":
            summary = textrank.summarize(text=original_text, n_sentences=n)
        else:
            summary = lexrank.lead_k_base_line(text=original_text, k=n)

    if original_text:
        print("\n-------------------------------------------------")
        print("----------------------TEXT-----------------------")
        print("-------------------------------------------------")
        print(original_text)

    print("\n-------------------------------------------------")
    print("---------------------SUMMARY---------------------")
    print("-------------------------------------------------")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
