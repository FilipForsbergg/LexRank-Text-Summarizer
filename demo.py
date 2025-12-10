import gradio as gr
from LexRank.LexRank import LexRank

# initiera en LexRank-instans en gång
lexrank = LexRank()

def summarize_interface(text, method, n_sentences):
    """
    Funktion som anropas av Gradio.
    """
    text = text.strip()
    if not text:
        return "Ingen text angiven."

    # välj metod
    if method == "LexRank":
        sentences = lexrank.summarize(text, n_sentences=int(n_sentences))
    else:  # Lead-k
        sentences = lexrank.lead_k_base_line(text, k=int(n_sentences))

    summary = " ".join(sentences)
    return summary


with gr.Blocks(title="LexRank Text Summarizer") as demo:
    gr.Markdown(
        """
        # LexRank Text Summarizer  
        Klistra in en artikel eller längre text, välj metod och antal meningar.
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Inmatningstext",
            placeholder="Klistra in en nyhetsartikel eller annan text här...",
            lines=15,
        )

    with gr.Row():
        method_input = gr.Radio(
            ["LexRank", "Lead-k"],
            value="LexRank",
            label="Metod",
        )
        n_sentences_input = gr.Slider(
            minimum=1,
            maximum=7,
            value=3,
            step=1,
            label="Antal meningar i sammanfattningen",
        )

    summarize_button = gr.Button("Sammanfatta")

    summary_output = gr.Textbox(
        label="Sammanfattning",
        lines=10,
    )

    summarize_button.click(
        summarize_interface,
        inputs=[text_input, method_input, n_sentences_input],
        outputs=summary_output,
    )

if __name__ == "__main__":
    demo.launch()