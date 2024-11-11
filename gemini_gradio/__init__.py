import os
from typing import Callable
import gradio as gr
import google.generativeai as genai

__version__ = "0.0.1"


def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(message, history, enable_search):
        inputs = preprocess(message, history, enable_search)
        is_gemini = model_name.startswith("gemini-")
        
        if is_gemini:
            genai.configure(api_key=api_key)
            
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            chat = model.start_chat(history=inputs.get("history", []))
            
            if inputs.get("enable_search"):
                response = chat.send_message(
                    message,
                    stream=True,
                    tools='google_search_retrieval'
                )
            else:
                response = chat.send_message(message, stream=True)
            
            response_text = ""
            for chunk in response:
                if chunk.text:
                    response_text += chunk.text
                    yield postprocess(response_text)

    return fn


def get_interface_args(pipeline, model_name: str):
    if pipeline == "chat":
        inputs = [gr.Checkbox(label="Enable Search", value=False)]
        outputs = None

        def preprocess(message, history, enable_search):
            is_gemini = model_name.startswith("gemini-")
            if is_gemini:
                gemini_history = []
                for user_msg, assistant_msg in history:
                    gemini_history.append({
                        "role": "user",
                        "parts": [{"text": user_msg}]
                    })
                    gemini_history.append({
                        "role": "model",
                        "parts": [{"text": assistant_msg}]
                    })
                return {
                    "history": gemini_history,
                    "message": message,
                    "enable_search": enable_search
                }
            else:
                messages = []
                for user_msg, assistant_msg in history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "user", "content": message})
                return {"messages": messages}

        postprocess = lambda x: x
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    return "chat"


def registry(
    name: str, 
    token: str | None = None, 
    examples: list | None = None,
    **kwargs
):
    env_key = "GEMINI_API_KEY"
    api_key = token or os.environ.get(env_key)
    if not api_key:
        raise ValueError(f"{env_key} environment variable is not set.")

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline, name)
    fn = get_fn(name, preprocess, postprocess, api_key)

    if examples:
        formatted_examples = [[example, False] for example in examples]
        kwargs["examples"] = formatted_examples

    if pipeline == "chat":
        interface = gr.ChatInterface(
            fn=fn,
            additional_inputs=inputs,
            **kwargs
        )
    else:
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
