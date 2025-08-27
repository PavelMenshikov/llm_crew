# app.py — Финальная версия без сломанной темы UI

import os
import re
import time
import json
import base64
import io
import requests
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from PIL import Image


print("Загружаю конфигурацию из .env...")
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Ключ OPENROUTER_API_KEY не найден в файле .env!")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

HISTORY_FILE = "chat_history.json"
DEFAULT_MAX_TOKENS = 4096


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                openai_history = json.load(f)
                display_pairs = []
                for i in range(0, len(openai_history), 2):
                    if i + 1 < len(openai_history):
                        u = openai_history[i]; a = openai_history[i + 1]
                        user_display = "[Изображение/Файл]"
                        if isinstance(u.get("content"), list):
                            text_part = next((part.get("text", "") for part in u.get("content") if isinstance(part, dict) and part.get("type") == "text"), "")
                            user_display = text_part or user_display
                        elif isinstance(u.get("content"), str):
                            user_display = u.get("content")
                        assistant_display = a.get("content")
                        display_pairs.append((user_display, assistant_display))
                print("История успешно загружена.")
                return display_pairs, openai_history
        except Exception as e:
            print(f"Ошибка при чтении файла истории: {e}. Начинаем новый диалог.")
    return [], []

def save_history(openai_history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(openai_history, f, ensure_ascii=False, indent=2)
    print(f"История диалога сохранена в '{HISTORY_FILE}'.")

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


display_pairs, chat_history_messages = load_history()
def display_pairs_to_messages(display_pairs):
    msgs = []
    if display_pairs:
        for user_disp, assistant_disp in display_pairs:
            msgs.append({"role": "user", "content": user_disp})
            msgs.append({"role": "assistant", "content": assistant_disp})
    return msgs
initial_gradio_history = display_pairs_to_messages(display_pairs)


FALLBACK_MODELS = [ "openrouter/auto", "qwen/qwen:7b-chat-v1.5:free", "meta-llama/llama-3-8b-instruct" ]

def extract_error_text(exc):
    try: return str(exc)
    except: return "Unknown error"
def parse_afforded_tokens_from_error(err_text):
    if not err_text: return None
    m = re.search(r"can only afford (\d+)", err_text)
    if not m: m = re.search(r"afford up to (\d+)", err_text)
    if not m: m = re.search(r"afford (\d+)", err_text)
    if m:
        try: return int(m.group(1))
        except: return None
    return None
def try_request_with_fallbacks(initial_model_choice, messages, max_tokens, fallback_models=FALLBACK_MODELS):
    last_error = None; model_used = None; response_text = None; did_trim = False
    attempts = []
    if initial_model_choice and "disabled" not in initial_model_choice: attempts.append(initial_model_choice)
    for m in fallback_models:
        if m not in attempts: attempts.append(m)
    if "openrouter/auto" not in attempts: attempts.append("openrouter/auto")
    for m in attempts:
        try:
            print(f"Попытка вызова модели: {m} (max_tokens={max_tokens})")
            resp = client.chat.completions.create(model=m, messages=messages, max_tokens=max_tokens)
            return m, resp.choices[0].message.content, None, did_trim
        except Exception as e:
            last_error = e
            err_text = str(e)
            print(f"Ошибка при вызове модели {m}: {err_text}")
            time.sleep(0.2)
    return None, None, last_error, did_trim
            

def respond(message, image_input, file_input, model_selector, chat_history_display):
    
    if model_selector == "disabled" or model_selector is None or not model_selector.strip():
        model_selector = "openrouter/auto"
    print(f"Запрос к модели (запрошено): {model_selector}")
    user_message_parts, user_display = [], message or ""
    if message and message.strip():
        user_message_parts.append({"type": "text", "text": message.strip()})
    if image_input:
        try:
            base64_image = image_to_base64(image_input)
            user_message_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            user_display = f"{user_display} [Скриншот]" if user_display else "[Скриншот]"
        except Exception as e: print(f"Ошибка обработки изображения: {e}")
    if file_input:
        try:
            with open(file_input.name, "r", encoding="utf-8") as f: file_content = f.read()
            wrapper = f"\n\n--- НАЧАЛО ФАЙЛА '{os.path.basename(file_input.name)}' ---\n{file_content}\n--- КОНЕЦ ФАЙЛА ---"
            if user_message_parts and user_message_parts[0]["type"] == "text": user_message_parts[0]["text"] += wrapper
            else: user_message_parts.insert(0, {"type": "text", "text": wrapper})
            user_display = f"{user_display} [Файл]" if user_display else f"[Файл: {os.path.basename(file_input.name)}]"
        except Exception as e: print(f"Ошибка чтения файла: {e}")
    if not user_message_parts: return chat_history_display, None, None, ""

    chat_history_messages.append({"role": "user", "content": user_message_parts})

    
    messages_to_send = display_pairs_to_messages(chat_history_display)
    messages_to_send.append({"role": "user", "content": user_display.strip()})
    
    model_used, response_text, error, trimmed = try_request_with_fallbacks(model_selector, messages_to_send[-11:], DEFAULT_MAX_TOKENS)

    if response_text is not None:
        labelled_response = f"**Отвечает:** `{model_used}`\n\n{response_text}"
        chat_history_messages.append({"role": "assistant", "content": response_text})
        chat_history_display.append({"role": "user", "content": user_display.strip()})
        chat_history_display.append({"role": "assistant", "content": labelled_response})
        save_history(chat_history_messages)
    else:
        err_text = str(error)
        chat_history_display.append({"role": "user", "content": user_display.strip()})
        chat_history_display.append({"role": "assistant", "content": f"**Ошибка:** Все попытки по моделям провалились. Последняя ошибка: {err_text}"})
        if chat_history_messages: chat_history_messages.pop()

    return chat_history_display, None, None, ""


with gr.Blocks(title="Локальный Командный Центр") as demo:
    gr.Markdown("# 🚀 Локальный Командный Центр")
    with gr.Row():
        with gr.Column(scale=4):
            
            chatbot = gr.Chatbot(value=initial_gradio_history, height=700, label="История диалога", show_copy_button=True, type="messages")
            msg_textbox = gr.Textbox(placeholder="Введите сообщение...", show_label=False, container=False)
            submit_button = gr.Button("Отправить", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Настройки")
            
            
            model_selector = gr.Dropdown(
                label="Выберите модель",
                choices=[
                    ("--- 👑 Топовые (Платные) ---", "disabled"),
                    ("⭐ Google: Gemini 1.5 Pro", "google/gemini-pro-1.5"),
                    ("🧠 Anthropic: Claude 3 Opus", "anthropic/claude-3-opus-20240229"),
                    ("🔥 OpenAI: GPT-4o", "openai/gpt-4o"),
                    
                    ("--- ✅ Сбалансированные (Дешевле) ---", "disabled"),
                    ("⚡ Anthropic: Claude 3 Sonnet", "anthropic/claude-3-sonnet-20240229"),
                    ("💨 Anthropic: Claude 3 Haiku", "anthropic/claude-3-haiku-20240307"),

                    ("--- 💸 Полностью Бесплатные ---", "disabled"),
                    ("🏆 Meta: Llama 3 8B Instruct", "meta-llama/llama-3-8b-instruct:free"),
                    ("💡 Mistral: Mixtral 8x7B Instruct", "mistralai/mixtral-8x7b-instruct:free"),
                    ("📖 Google: Gemma 7B Instruct", "google/gemma-7b-it:free"),
                    ("💬 Alibaba: Qwen 1.5 7B Chat", "qwen/qwen:7b-chat-v1.5:free"),
                    ("✨ Auto Router (OpenRouter)", "openrouter/auto"),
                ],
                value="openrouter/auto"
            )
            image_box = gr.Image(sources=["upload", "clipboard"], type="pil", label="Вставьте (Ctrl+V) или загрузите скриншот")
            file_box = gr.File(label="Загрузите текстовый документ")

            def clear_all():
                global chat_history_messages; chat_history_messages = [];
                if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
                return [], None, None, ""

            clear_btn = gr.Button("🗑️ Начать новый диалог")
            clear_btn.click(clear_all, [], [chatbot, image_box, file_box, msg_textbox])

    inputs_for_respond = [msg_textbox, image_box, file_box, model_selector, chatbot]
    outputs_for_respond = [chatbot, image_box, file_box, msg_textbox]

    submit_button.click(respond, inputs=inputs_for_respond, outputs=outputs_for_respond)
    msg_textbox.submit(respond, inputs=inputs_for_respond, outputs=outputs_for_respond)

# --- Запуск ---
if __name__ == "__main__":
    print("Запуск веб-интерфейса...")
    demo.launch()