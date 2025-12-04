from LexRank.LexRank import LexRank, CNNDailyMailCorpus
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='LexRank text summarization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', type=str, help='Text file to summarize')
    parser.add_argument('--n_sentences', '-n', type=int, default=3, help='Amount of output sentences')
    group.add_argument('--text', '-t', type=str, help='Text to summarize')

    args = parser.parse_args()
    lexrank = LexRank()

    original_text = None
    if file := args.file.strip():
        if not os.path.exists(file):
            raise ValueError("Filepath doesn't exist")
        with open(file, 'r', encoding="utf-8") as f:
            text_content = f.read()
        summary = lexrank.summarize(text=text_content, n_sentences=int(args.n_sentences))
    elif text := args.text:
        summary = lexrank.summarize(text=text, n_sentences=int(args.n_sentences))
    else:
        #Sample a CNN article
        cnn = CNNDailyMailCorpus()
        original_text = cnn.load_random_article()
        summary = lexrank.summarize(text=original_text, n_sentences=int(args.n_sentences))

    if original_text:
        print("-------------------------------------------------")
        print("----------------------TEXT----------------------")
        print("-------------------------------------------------", end='\n')
        print(original_text)

    print("-------------------------------------------------")
    print("---------------------SUMMARY---------------------")
    print("-------------------------------------------------", end='\n')
    print("".join(summary))

if __name__ == "__main__":
    main()