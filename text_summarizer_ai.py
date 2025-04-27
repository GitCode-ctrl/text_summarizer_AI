
from transformers import pipeline     #it connects a model to a task like summarization, translation, etc.
import gradio as gr                   #Python library that makes it super easy to create web apps

summarizer = pipeline("summarization")  #create a summarizer tool by calling pipeline("summarization") Behind the scenes, it downloads a powerful pre-trained AI model like BART or T5 that knows how to summarize text!

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

iface = gr.Interface(
    fn=summarize_text,
    inputs="text",
    outputs="text",
    title="📜 Text Summarizer AI",
    description="Paste any article or paragraph, and this AI will summarize it for you!"
)

iface.launch(share = True)




