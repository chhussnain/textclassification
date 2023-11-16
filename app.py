!pip install --upgrade pip

import gradio as gr
from transformers import pipeline

model_checkpoint = "hagara/roberta-large-2"

# Load the text classification pipeline
pipe = pipeline("text-classification", model=model_checkpoint)

def classify_text(text, question):
    result = pipe(question, text)
    if result[0]['label'] == 'LABEL_0':
        result[0]['label'] = 'yes'
    elif result[0]['label'] == 'LABEL_1':
        result[0]['label'] = 'no'
    return result[0]['label'], result[0]['score']

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs=["text", "text"],
    outputs=["text", "number"],
    layout="vertical",
    live=True,
    title="Get yes/no answer for your medical question",
    description="Predict if a statement is true or false."
)

# Launch the Gradio interface
iface.launch()
