import os
from typing import Callable
import gradio as gr
import google.generativeai as genai
import base64
import json
import numpy as np
import websockets.sync.client
from gradio_webrtc import StreamHandler, WebRTC, get_twilio_turn_credentials
import cv2
import PIL.Image
import io

__version__ = "0.0.3"


class GeminiConfig:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.host = "generativelanguage.googleapis.com"
        self.model = "models/gemini-2.0-flash-exp"
        self.ws_url = f"wss://{self.host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"


class AudioProcessor:
    @staticmethod
    def encode_audio(data, sample_rate):
        encoded = base64.b64encode(data.tobytes()).decode("UTF-8")
        return {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": f"audio/pcm;rate={sample_rate}",
                        "data": encoded,
                    }
                ],
            },
        }

    @staticmethod
    def process_audio_response(data):
        audio_data = base64.b64decode(data)
        return np.frombuffer(audio_data, dtype=np.int16)


def detection(frame, conf_threshold=0.3):
    """Process video frame."""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL Image
        pil_image = PIL.Image.fromarray(image_rgb)
        pil_image.thumbnail([1024, 1024])
        
        # Convert back to numpy array
        processed_frame = np.array(pil_image)
        
        # Convert back to BGR for OpenCV
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        return processed_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame


class GeminiHandler(StreamHandler):
    def __init__(self, expected_layout="mono", output_sample_rate=24000, output_frame_size=480) -> None:
        super().__init__(expected_layout, output_sample_rate, output_frame_size, input_sample_rate=24000)
        self.config = GeminiConfig()
        self.ws = None
        self.all_output_data = None
        self.audio_processor = AudioProcessor()
        self.current_frame = None

    def copy(self):
        handler = GeminiHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )
        return handler

    def _initialize_websocket(self):
        try:
            self.ws = websockets.sync.client.connect(self.config.ws_url, timeout=30)
            initial_request = {
                "setup": {
                    "model": self.config.model,
                }
            }
            self.ws.send(json.dumps(initial_request))
            setup_response = json.loads(self.ws.recv())
            print(f"Setup response: {setup_response}")
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket connection failed: {str(e)}")
            self.ws = None
        except Exception as e:
            print(f"Setup failed: {str(e)}")
            self.ws = None

    def process_video_frame(self, frame):
        self.current_frame = frame
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = base64.b64encode(buffer).decode('utf-8')
        return image_data

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        try:
            if not self.ws:
                self._initialize_websocket()

            _, array = frame
            array = array.squeeze()
            
            audio_data = self.audio_processor.encode_audio(array, self.output_sample_rate)
            
            message = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": f"audio/pcm;rate={self.output_sample_rate}",
                            "data": audio_data["realtimeInput"]["mediaChunks"][0]["data"],
                        }
                    ],
                }
            }
            
            if self.current_frame is not None:
                image_data = self.process_video_frame(self.current_frame)
                message["realtimeInput"]["mediaChunks"].append({
                    "mimeType": "image/jpeg",
                    "data": image_data
                })

            self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"Error in receive: {str(e)}")
            if self.ws:
                self.ws.close()
            self.ws = None

    def _process_server_content(self, content):
        for part in content.get("parts", []):
            data = part.get("inlineData", {}).get("data", "")
            if data:
                audio_array = self.audio_processor.process_audio_response(data)
                if self.all_output_data is None:
                    self.all_output_data = audio_array
                else:
                    self.all_output_data = np.concatenate((self.all_output_data, audio_array))

                while self.all_output_data.shape[-1] >= self.output_frame_size:
                    yield (self.output_sample_rate, self.all_output_data[: self.output_frame_size].reshape(1, -1))
                    self.all_output_data = self.all_output_data[self.output_frame_size :]

    def generator(self):
        while True:
            if not self.ws:
                print("WebSocket not connected")
                yield None
                continue

            try:
                message = self.ws.recv(timeout=5)
                msg = json.loads(message)

                if "serverContent" in msg:
                    content = msg["serverContent"].get("modelTurn", {})
                    yield from self._process_server_content(content)
            except TimeoutError:
                print("Timeout waiting for server response")
                yield None
            except Exception as e:
                print(f"Error in generator: {str(e)}")
                yield None

    def emit(self) -> tuple[int, np.ndarray] | None:
        if not self.ws:
            return None
        if not hasattr(self, "_generator"):
            self._generator = self.generator()
        try:
            return next(self._generator)
        except StopIteration:
            self.reset()
            return None

    def reset(self) -> None:
        if hasattr(self, "_generator"):
            delattr(self, "_generator")
        self.all_output_data = None

    def shutdown(self) -> None:
        if self.ws:
            self.ws.close()

    def check_connection(self):
        try:
            if not self.ws or self.ws.closed:
                self._initialize_websocket()
            return True
        except Exception as e:
            print(f"Connection check failed: {str(e)}")
            return False


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
    enable_voice: bool = False,
    enable_video: bool = False,
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
        if enable_voice or enable_video:
            interface = gr.Blocks()
            with interface:
                gr.HTML(
                    """
                    <div style='text-align: center'>
                        <h1>Gemini Chat</h1>
                    </div>
                    """
                )
                
                gemini_handler = GeminiHandler()
                
                with gr.Row():
                    with gr.Column(scale=1):
                        if enable_video:
                            video = WebRTC(
                                label="Stream",
                                mode="send-receive",
                                modality="video",
                                rtc_configuration=get_twilio_turn_credentials()
                            )

                        if enable_voice:
                            audio = WebRTC(
                                label="Voice Chat",
                                modality="audio",
                                mode="send-receive",
                                rtc_configuration=get_twilio_turn_credentials(),
                            )

                if enable_video:
                    video.stream(
                        fn=lambda frame: (gemini_handler.process_video_frame(detection(frame)), frame)[1],
                        inputs=[video],
                        outputs=[video],
                        time_limit=90,
                        concurrency_limit=10
                    )

                if enable_voice:
                    audio.stream(
                        gemini_handler,
                        inputs=[audio], 
                        outputs=[audio], 
                        time_limit=90, 
                        concurrency_limit=10
                    )
        else:
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
