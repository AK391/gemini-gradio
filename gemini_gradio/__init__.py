import os
from typing import Callable
import gradio as gr
import google.generativeai as genai

__version__ = "0.0.2"


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
                    inputs["message"],
                    stream=True,
                    tools='google_search_retrieval'
                )
            else:
                response = chat.send_message(inputs["message"], stream=True)
            
            response_text = ""
            for chunk in response:
                if chunk.text:
                    response_text += chunk.text
                    yield {"role": "assistant", "content": response_text}

    return fn


def get_interface_args(pipeline, model_name: str):
    if pipeline == "chat":
        inputs = [gr.Checkbox(label="Enable Search", value=False)]
        outputs = None

        def preprocess(message, history, enable_search):
            is_gemini = model_name.startswith("gemini-")
            if is_gemini:
                # Handle multimodal input
                if isinstance(message, dict):
                    parts = []
                    if message.get("text"):
                        parts.append({"text": message["text"]})
                    if message.get("files"):
                        for file in message["files"]:
                            # Determine file type and handle accordingly
                            if isinstance(file, str):  # If it's a file path
                                mime_type = None
                                if file.lower().endswith('.pdf'):
                                    mime_type = "application/pdf"
                                elif file.lower().endswith('.txt'):
                                    mime_type = "text/plain"
                                elif file.lower().endswith('.html'):
                                    mime_type = "text/html"
                                elif file.lower().endswith('.md'):
                                    mime_type = "text/md"
                                elif file.lower().endswith('.csv'):
                                    mime_type = "text/csv"
                                elif file.lower().endswith(('.js', '.javascript')):
                                    mime_type = "application/x-javascript"
                                elif file.lower().endswith('.py'):
                                    mime_type = "application/x-python"
                                
                                if mime_type:
                                    try:
                                        uploaded_file = genai.upload_file(file)
                                        parts.append(uploaded_file)
                                    except Exception as e:
                                        print(f"Error uploading file: {e}")
                                else:
                                    with open(file, "rb") as f:
                                        image_data = f.read()
                                        import base64
                                        image_data = base64.b64encode(image_data).decode()
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": "image/jpeg",
                                                "data": image_data
                                            }
                                        })
                            else:  # If it's binary data, treat as image
                                import base64
                                image_data = base64.b64encode(file).decode()
                                parts.append({
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_data
                                    }
                                })
                    message_parts = parts
                else:
                    message_parts = [{"text": message}]

                # Process history
                gemini_history = []
                for entry in history:
                    # Handle different history formats
                    if isinstance(entry, (list, tuple)):
                        user_msg, assistant_msg = entry
                    else:
                        # If it's a dict with role/content format
                        if entry.get("role") == "user":
                            user_msg = entry.get("content")
                            continue  # Skip to next iteration to get assistant message
                        elif entry.get("role") == "assistant":
                            assistant_msg = entry.get("content")
                            continue  # Skip to next iteration
                        else:
                            continue  # Skip unknown roles

                    # Process user message
                    if isinstance(user_msg, dict):
                        parts = []
                        if user_msg.get("text"):
                            parts.append({"text": user_msg["text"]})
                        if user_msg.get("files"):
                            for file in user_msg["files"]:
                                if isinstance(file, str):
                                    mime_type = None
                                    if file.lower().endswith('.pdf'):
                                        mime_type = "application/pdf"
                                    # ... (same mime type checks as before)
                                    
                                    if mime_type:
                                        try:
                                            uploaded_file = genai.upload_file(file)
                                            parts.append(uploaded_file)
                                        except Exception as e:
                                            print(f"Error uploading file in history: {e}")
                                    else:
                                        with open(file, "rb") as f:
                                            image_data = f.read()
                                            import base64
                                            image_data = base64.b64encode(image_data).decode()
                                            parts.append({
                                                "inline_data": {
                                                    "mime_type": "image/jpeg",
                                                    "data": image_data
                                                }
                                            })
                                else:
                                    import base64
                                    image_data = base64.b64encode(file).decode()
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": image_data
                                        }
                                    })
                        gemini_history.append({
                            "role": "user",
                            "parts": parts
                        })
                    else:
                        gemini_history.append({
                            "role": "user",
                            "parts": [{"text": str(user_msg)}]
                        })
                    
                    # Process assistant message
                    gemini_history.append({
                        "role": "model",
                        "parts": [{"text": str(assistant_msg)}]
                    })
                
                return {
                    "history": gemini_history,
                    "message": message_parts,
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
            multimodal=True,
            type="messages",
            **kwargs
        )
    else:
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
