import gradio as gr
import gemini_gradio

with gr.Blocks() as demo:
    with gr.Tab("Gemini 1.5 Pro"):
        gr.load('gemini-1.5-flash', src=gemini_gradio.registry)
    with gr.Tab("Gemini 1.5 Flash"):
        gr.load('gemini-1.5-pro', src=gemini_gradio.registry)

demo.launch()