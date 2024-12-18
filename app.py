import gradio as gr
import gemini_gradio

gr.load(
    name='gemini-2.0-flash-exp',
    src=gemini_gradio.registry,
    enable_video=True,
    enable_voice=True
).launch()
