import sqlite3, random, asyncio, os, requests
from io import BytesIO
from gtts import gTTS
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. THE KEYS & CONFIG (FROM .env FILE)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"
PROXIES = {"http": "http://proxy.server:3128", "https": "http://proxy.server:3128"}

# Verify that all required keys are loaded
assert TELEGRAM_TOKEN, "ERROR: TELEGRAM_TOKEN not found in .env file"
assert OPENROUTER_KEY, "ERROR: OPENROUTER_KEY not found in .env file"
assert HF_TOKEN, "ERROR: HF_TOKEN not found in .env file"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

# ==========================================
# 2. INTERNAL ENGINES (Voice & Eyes)
# ==========================================
def generate_voice_sync(text):
    try:
        # 2026 Master-Apprentice Fix: Strip roleplay symbols before speaking
        clean_text = text.replace("*", "").replace('"', "")
        tts = gTTS(text=clean_text, lang='en', tld='co.uk')
        file_path = "arya_voice.mp3"
        tts.save(file_path)
        return file_path
    except Exception as e:
        print(f"Cloud Voice Error: {e}")
        return None

def generate_image_sync(prompt):
    # This locks her look regardless of the user's request
    MANDATORY_LOOK = "24-year-old woman, sharp chin-length dark hair bob, expressive dark eyes, natural realistic skin texture, athletic build"

    # We force the style to be 'cinematic photo' to avoid sketches
    full_prompt = f"Cinematic photo of {MANDATORY_LOOK}, {prompt}, high quality, 8k, sharp focus, vibrant colors"
    negative = "sketch, black and white, drawing, cartoon, extra fingers, deformed, blurry, plastic skin"

    API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Adding a higher guidance scale helps the AI follow your prompt better
    payload = {
        "inputs": full_prompt,
        "parameters": {"negative_prompt": negative, "guidance_scale": 9.0}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, proxies=PROXIES, timeout=60)
        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img_data.seek(0)
            img_data.name = 'arya_capture.jpg'
            return img_data
    except Exception as e:
        print(f"Image API Error: {e}")
    return None

# ==========================================
# 3. DATABASE & MEMORY (Long-Term Retrieval)
# ==========================================
def init_db():
    conn = sqlite3.connect('arya_memory.db')
    conn.cursor().execute('CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

def search_old_memories(query, limit=3):
    try:
        conn = sqlite3.connect('arya_memory.db')
        # Scans history for keywords like 'Burnaby' or 'Meeting'
        rows = conn.cursor().execute("SELECT message FROM chat_history WHERE message LIKE ? AND sender = 'User' ORDER BY timestamp DESC LIMIT ?", ('%' + query + '%', limit)).fetchall()
        conn.close()
        return " | ".join([r[0] for r in rows])
    except:
        return ""

def save_to_memory(sender, text):
    conn = sqlite3.connect('arya_memory.db')
    conn.cursor().execute("INSERT INTO chat_history (sender, message) VALUES (?, ?)", (sender, text))
    conn.commit()
    conn.close()

# ==========================================
# 4. HANDLERS
# ==========================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id
    save_to_memory("User", user_text)
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Inject Long-Term context if keywords match
    past_context = ""
    if any(k in user_text.lower() for k in ["remember", "apartment", "burnaby", "data pilot"]):
        past_context = f"\nRELEVANT PAST INFO: {search_old_memories(user_text[:10])}"

    with open("soul.txt", "r") as f: soul_content = f.read()

    # Rebuild history
    conn = sqlite3.connect('arya_memory.db')
    rows = conn.cursor().execute("SELECT sender, message FROM chat_history ORDER BY timestamp DESC LIMIT 10").fetchall()
    conn.close()

    messages = [{"role": "system", "content": soul_content + past_context}]
    for r in reversed(rows):
        messages.append({"role": "user" if r[0] == "User" else "assistant", "content": r[1]})

    try:
        response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.9)
        arya_reply = response.choices[0].message.content or "..."
        save_to_memory("Arya", arya_reply)

        for part in arya_reply.split('\n\n')[:3]:
            if "IMAGE_PROMPT:" in part:
                asyncio.create_task(send_photo_task(context, chat_id, part.split("IMAGE_PROMPT:")[1].strip()))
                continue

            if random.random() < 0.25:
                await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
                audio = await asyncio.to_thread(generate_voice_sync, part.strip())
                if audio: await update.message.reply_voice(voice=open(audio, 'rb'))
                else: await update.message.reply_text(part.strip())
            else:
                await update.message.reply_text(part.strip())
    except Exception as e:
        print(f"Brain Error: {e}")

async def send_photo_task(context, chat_id, prompt):
    photo = await asyncio.to_thread(generate_image_sync, prompt)
    if photo: await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="for you... 😉")

# ==========================================
# 5. START
# ==========================================
if __name__ == '__main__':
    init_db()
    print("Arya 2.8 Memory Monolith Waking up...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()