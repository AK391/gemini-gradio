import gradio as gr
import gemini_gradio

gr.load(
    name='gemini-1.5-pro',
    src=gemini_gradio.registry,
    title='Gemini Pro Integration',
    description="Chat with Google's Gemini 1.5 Pro model.",
    examples=["Explain quantum gravity to a 5-year old.", "Write a creative story about a magical library."]
).launch()