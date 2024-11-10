import gradio as gr
import gemini_gradio

gr.load(
    name='gemini-1.5-flash',
    src=gemini_gradio.registry,
).launch()