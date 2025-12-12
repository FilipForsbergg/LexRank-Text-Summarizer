import gradio as gr
from LexTexRank.LexTexRank import LexRank, TextRank, CNNDailyMailCorpus

lexrank = LexRank()
textrank = TextRank()

corpus = CNNDailyMailCorpus(amount=1000)

def summarize_interface(text, method, n_sentences):
    text = text.strip()
    if not text:
        return "Ingen text angiven."

    n = int(n_sentences)

    if method == "LexRank":
        sentences = lexrank.summarize(text, n_sentences=n)
    elif method == "TextRank":
        sentences = textrank.summarize(text, n_sentences=n)
    else:  # Lead-k
        sentences = lexrank.lead_k_base_line(text, k=n)

    summary = " ".join(sentences)
    return summary


def load_example_article():
    """
    Hämtar en slumpad artikel från CNN/DailyMail-korpuset.
    Fyller textrutan i UIn
    """
    article_text = corpus.load_random_article()
    return article_text


with gr.Blocks(title="LexRank Text Summarizer") as demo:
    gr.Markdown(
        """
        # Lex/Text-Rank Text Summarizer  
        Klistra in en artikel eller använd en slumpad CNN/DailyMail-artikel.
        Välj metod och antal meningar, och klicka på **Sammanfatta**.
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Inmatningstext",
            placeholder="Klistra in en nyhetsartikel eller annan text här...",
            lines=18,
        )

    with gr.Row():
        method_input = gr.Radio(
            ["LexRank", "TextRank", "Lead-k"],
            value="LexRank",
            label="Metod",
        )
        n_sentences_input = gr.Slider(
            minimum=1,
            maximum=20,
            value=3,
            step=1,
            label="Antal meningar i sammanfattningen",
        )

    with gr.Row():
        summarize_button = gr.Button("Sammanfatta")
        example_button = gr.Button("Ladda exempelartikel (CNN/DailyMail)")

    summary_output = gr.Textbox(
        label="Sammanfattning",
        lines=12,
    )

    #Koppla knapparna
    summarize_button.click(
        summarize_interface,
        inputs=[text_input, method_input, n_sentences_input],
        outputs=summary_output,
    )

    example_button.click(
        load_example_article,
        inputs=[],
        outputs=text_input,   # fyller textfältet
    )

if __name__ == "__main__":
    demo.launch()