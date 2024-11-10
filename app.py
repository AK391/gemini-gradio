import gradio as gr
import gemini_gradio

gr.load(
    name='gpt-4-turbo',
    src=gemini_gradio.registry,
).launch()