"""
ARYA 4.0 - PRODUCTION FINAL VERSION
Premium AI Companion Service
Production-ready with proper logging and API robustness

Features:
- Professional logging to stdout (not stderr)
- Retry logic for image generation
- Timezone-aware datetime handling
- 5 organized sections for easy maintenance
"""

import asyncio
import os
import logging
import json
import random
import requests
import re
import urllib.parse
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

from gtts import gTTS
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackContext
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client

# =====================================================
# SECTION 1: CONFIG & INITIALIZATION
# =====================================================

# Load environment variables
load_dotenv()

# Configure logging to output to STDOUT (not stderr)
import sys
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]  # ✅ FIXED: Explicitly use stdout
)
logger = logging.getLogger(__name__)

# Suppress verbose library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Unbuffer output for Railway
sys.stdout.reconfigure(line_buffering=True)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== TELEGRAM CONFIGURATION =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# ===== LLM (BRAIN) CONFIGURATION =====
BRAIN_PROVIDER = "openrouter"
BRAIN_API_KEY = os.getenv("OPENROUTER_KEY")
BRAIN_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "deepseek/deepseek-v3.2"

# ===== VOICE TRANSCRIPTION CONFIGURATION =====
TRANSCRIPTION_PROVIDER = "groq"
TRANSCRIPTION_API_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPTION_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3"

# ===== VOICE GENERATION (TTS) CONFIGURATION =====
TTS_PROVIDER = "gtts"

# ===== IMAGE GENERATION CONFIGURATION =====
IMAGE_PROVIDER = "pollinations"
IMAGE_API_KEY = os.getenv("POLLINATIONS_API_KEY")
IMAGE_BASE_URL = "https://image.pollinations.ai/prompt"
IMAGE_MODEL = "flux"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
# Simplified mandatory look to reduce API failures
IMAGE_MANDATORY_LOOK = (
    "24-year-old woman, dark hair bob, dark eyes, athletic build. "
    "Casual wear. Natural lighting, photography. Real person."
)

# ===== DATABASE CONFIGURATION =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ===== PERSONALITY FILES =====
PERSONALITY_FILES = {
    'female': 'soul_female.txt',
    'male': 'soul_male.txt',
    'non-binary': 'soul_nb.txt'
}

# ===== ERROR MESSAGES (HUMANISTIC) =====
ERROR_MESSAGES = {
    "general": [
        "ugh my head hurts rn 😅",
        "i'm having a weird moment, can we try again?",
        "sorry brain fog happening, ask me later",
        "lowkey not thinking straight right now",
        "i'm kinda out of it, can we try again?",
        "head's all fuzzy, give me a sec",
        "idk what's wrong with me rn honestly",
        "my brain needs a coffee break lol",
        "having a weird technical hiccup, sorry",
        "signal's being dumb, try again?",
    ],
    "voice": [
        "couldn't quite hear that, sorry",
        "my ears are broken rn lol",
        "couldn't catch that, can you repeat?",
        "you're breaking up i think?",
        "didn't get that one, try again?",
    ],
    "image": [
        "can't find my camera rn 😅",
        "picture's not loading, try again?",
        "my phone's being weird, hold on",
        "camera's acting up, sorry babe",
        "can't get a good shot rn",
    ]
}

# ===== RESPONSE CLEANING PATTERNS =====
THINKING_PATTERNS = [
    "alright,", "first,", "she should", "the tone should", "this is",
    "since", "arya needs", "arya's", "the user", "let me", "i need to",
    "the response", "she needs", "this means", "so she", "aiming for",
    "prompt should", "visual prompt", "arya's reply", "wrapping up",
    "acknowledging", "the vibe", "respond to", "focus on"
]

# ===== ONBOARDING QUESTIONS =====
ONBOARDING_QUESTIONS = [
    {"id": 1, "question": "what's your name btw? i don't think i asked 😅", "field": "user_name", "day": 1},
    {"id": 2, "question": "so what do you do for work? or are you studying?", "field": "user_job", "day": 2},
    {"id": 3, "question": "what kinds of things do you like to do? hobbies and stuff?", "field": "user_hobbies", "day": 3},
]

# ===== VERIFY AND INITIALIZE =====
def verify_config():
    """Verify all required API keys are present"""
    required_keys = {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "OPENROUTER_KEY": BRAIN_API_KEY,
        "GROQ_API_KEY": TRANSCRIPTION_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
        "POLLINATIONS_API_KEY": IMAGE_API_KEY,
    }
    
    for key_name, key_value in required_keys.items():
        if not key_value:
            logger.error(f"❌ Missing: {key_name}")
            raise ValueError(f"Missing environment variable: {key_name}")
    
    logger.info("✅ All API keys verified!")

verify_config()

# Initialize API clients
brain_client = OpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
transcription_client = OpenAI(base_url=TRANSCRIPTION_BASE_URL, api_key=TRANSCRIPTION_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logger.info("✅ All API clients initialized!")


# =====================================================
# SECTION 2: DATABASE FUNCTIONS
# =====================================================

# ===== USER MANAGEMENT =====
def get_or_create_user(telegram_id: int) -> Optional[str]:
    """Get existing user or create new one"""
    try:
        result = supabase.table("users").select("id").eq("telegram_id", telegram_id).execute()
        if result.data:
            return result.data[0]["id"]
        
        user_result = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        return user_result.data[0]["id"] if user_result.data else None
    except Exception as e:
        logger.error(f"Error in get_or_create_user: {e}")
        return None

def get_user_profile(user_id: str) -> Optional[Dict]:
    """Fetch user profile from database"""
    try:
        result = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        return None

def update_user_profile(user_id: str, updates: Dict) -> bool:
    """Update user profile with new data"""
    try:
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("user_profiles").update(updates).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        return False

# ===== CONVERSATION HISTORY =====
def save_to_memory(user_id: str, sender: str, message: str) -> bool:
    """Save message to conversation history"""
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": sender,
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error saving to memory: {e}")
        return False

def get_conversation_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve conversation history for context"""
    try:
        result = supabase.table("conversations").select("*").eq("user_id", user_id).order(
            "created_at", desc=True
        ).limit(limit).execute()
        return list(reversed(result.data)) if result.data else []
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []

# ===== ONBOARDING & VOICE TRACKING =====
def should_ask_onboarding_question(user_id: str) -> Optional[Dict]:
    """Check if user should be asked next onboarding question"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return ONBOARDING_QUESTIONS[0]
        
        questions_asked = profile.get("questions_asked_today", 0)
        if questions_asked < len(ONBOARDING_QUESTIONS):
            return ONBOARDING_QUESTIONS[questions_asked]
        return None
    except Exception as e:
        logger.error(f"Error checking onboarding: {e}")
        return None

def can_send_voice_today(user_id: str) -> bool:
    """Check if user hasn't exceeded daily voice limit"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True
        
        last_voice = profile.get("last_voice_sent_at")
        if not last_voice:
            return True
        
        last_voice_time = datetime.fromisoformat(last_voice)
        if last_voice_time.tzinfo is None:
            last_voice_time = last_voice_time.replace(tzinfo=timezone.utc)
        
        last_voice_date = last_voice_time.date()
        return last_voice_date < datetime.now(timezone.utc).date()
    except Exception as e:
        logger.error(f"Error checking voice limit: {e}")
        return True

def mark_voice_sent(user_id: str) -> bool:
    """Mark that voice was sent today"""
    return update_user_profile(user_id, {"last_voice_sent_at": datetime.now(timezone.utc).isoformat()})

def should_send_checkin(user_id: str) -> bool:
    """Check if 24+ hours since last message"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return False
        
        last_msg = profile.get("last_message_from_user")
        if not last_msg:
            return False
        
        last_msg_time = datetime.fromisoformat(last_msg)
        if last_msg_time.tzinfo is None:
            last_msg_time = last_msg_time.replace(tzinfo=timezone.utc)
        
        now_utc = datetime.now(timezone.utc)
        return now_utc - last_msg_time > timedelta(hours=24)
    except Exception as e:
        logger.error(f"Error checking checkin: {e}")
        return False

def mark_checkin_sent(user_id: str) -> bool:
    """Mark that checkin was sent"""
    return update_user_profile(user_id, {"last_checkin_sent": datetime.now(timezone.utc).isoformat()})


# =====================================================
# SECTION 3: CORE LOGIC (LLM, IMAGES, VOICE)
# =====================================================

# ===== PERSONALITY & PROMPTING =====
def load_personality(gender: str = 'female') -> str:
    """Load personality definition from file"""
    try:
        filename = PERSONALITY_FILES.get(gender, 'soul_female.txt')
        filepath = os.path.join(SCRIPT_DIR, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Personality file not found: {filepath}")
        return "You are Arya, a 24-year-old woman with a vivid inner life."
    except Exception as e:
        logger.error(f"Error loading personality: {e}")
        return "You are Arya, a 24-year-old woman with a vivid inner life."

def clean_response(text: str) -> str:
    """Remove LLM internal thinking, asterisks, and quotes from response"""
    # Remove asterisk actions like *laughs*, *pauses*, etc
    text = re.sub(r'\*[^*]*\*', '', text)
    
    # Remove opening/closing quotes from entire response
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        skip = False
        for pattern in THINKING_PATTERNS:
            if pattern in line.lower():
                skip = True
                break
        
        if len(line) > 200 and any(word in line.lower() for word in ["should", "needs", "arya", "first"]):
            skip = True
        
        if not skip and line.strip():
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    if not result:
        result = text.strip()
    
    return result

def split_into_messages(text: str) -> List[str]:
    """Split long response into 2-3 shorter messages for natural feel"""
    text = text.strip()
    
    # If short, send as one message
    if len(text) <= 150:
        return [text]
    
    # Split by sentences
    sentences = text.split('. ')
    
    messages = []
    current = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith('.'):
            sentence += '.'
        
        # If adding this would be too long, save current and start new
        if len(current) + len(sentence) > 180:
            if current:
                messages.append(current.strip())
            current = sentence
        else:
            if current:
                current += " " + sentence
            else:
                current = sentence
    
    if current:
        messages.append(current.strip())
    
    # Return max 3 messages
    return messages[:3]

def get_random_error(error_type: str = "general") -> str:
    """Get random humanistic error message"""
    messages = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general"])
    return random.choice(messages)

# ===== LLM RESPONSE GENERATION =====
def generate_response(user_id: str, user_message: str, personality: str = 'female') -> Optional[str]:
    """Generate AI response using LLM"""
    try:
        logger.info(f"Generating response for user {user_id}")
        
        soul_content = load_personality(personality)
        history = get_conversation_history(user_id, limit=10)
        
        profile = get_user_profile(user_id)
        user_context = ""
        if profile:
            if profile.get("user_name"):
                user_context += f"\nUSER'S NAME: {profile['user_name']}"
            if profile.get("user_job"):
                user_context += f"\nUSER'S JOB: {profile['user_job']}"
            if profile.get("user_hobbies"):
                user_context += f"\nUSER'S HOBBIES: {profile['user_hobbies']}"
        
        messages = [{"role": "system", "content": soul_content + user_context}]
        for record in history:
            role = "user" if record["sender"] == "user" else "assistant"
            msg = record["message"].replace("[voice] ", "")
            messages.append({"role": role, "content": msg})
        
        messages.append({"role": "user", "content": user_message})
        
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.9
        )
        
        reply = response.choices[0].message.content or "..."
        reply = clean_response(reply)
        
        logger.info(f"Response generated: {reply[:50]}...")
        return reply
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

# ===== IMAGE GENERATION WITH RETRY =====
def generate_image_sync(prompt: str, retry_count: int = 0) -> Optional[BytesIO]:
    """Generate image of Arya with retry logic"""
    try:
        logger.info(f"Starting image generation (attempt {retry_count + 1})")
        
        # Simplify prompt on retries to reduce API failures
        if retry_count > 0:
            full_prompt = f"woman with dark hair, casual wear, natural lighting"
        else:
            full_prompt = f"{IMAGE_MANDATORY_LOOK}. {prompt}"
        
        params = {
            "prompt": full_prompt,
            "model": IMAGE_MODEL,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        }
        
        url = f"{IMAGE_BASE_URL}?{urllib.parse.urlencode(params)}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ Image generated successfully")
            return BytesIO(response.content)
        elif response.status_code >= 500 and retry_count < 1:
            # Retry once on server errors (5xx)
            logger.warning(f"Image API returned {response.status_code}, retrying...")
            return generate_image_sync(prompt, retry_count + 1)
        else:
            logger.error(f"Image API returned {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# ===== VOICE TRANSCRIPTION =====
def transcribe_voice_sync(voice_bytes: bytes) -> Optional[str]:
    """Transcribe voice to text using Groq"""
    try:
        logger.info("Starting voice transcription")
        
        response = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=("audio.ogg", voice_bytes),
        )
        
        text = response.text
        logger.info(f"Transcribed: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"Error transcribing voice: {e}")
        return None

# ===== VOICE GENERATION (TTS) =====
def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """Convert text to speech using gTTS"""
    try:
        logger.info("Starting voice generation")
        
        tts = gTTS(text=text, lang='en', slow=False)
        voice_buffer = BytesIO()
        tts.write_to_fp(voice_buffer)
        voice_buffer.seek(0)
        
        logger.info("✅ Voice generated successfully")
        return voice_buffer
    except Exception as e:
        logger.error(f"Error generating voice: {e}")
        return None


# =====================================================
# SECTION 4: MESSAGE HANDLERS (TELEGRAM)
# =====================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    user_message = update.message.text
    
    logger.info(f"Message from user {user_id_telegram}: {user_message[:50]}...")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error("general"))
        return
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        save_to_memory(user_id, "user", user_message)
        update_user_profile(user_id, {"last_message_from_user": datetime.now(timezone.utc).isoformat()})
        
        profile = get_user_profile(user_id)
        personality = profile.get("personality_choice", "female") if profile else "female"
        
        arya_reply = generate_response(user_id, user_message, personality)
        if not arya_reply:
            await update.message.reply_text(get_random_error("general"))
            return
        
        save_to_memory(user_id, "arya", arya_reply)
        
        # Split into 2-3 messages if too long
        messages = split_into_messages(arya_reply)
        for msg in messages:
            await update.message.reply_text(msg)
            await asyncio.sleep(0.3)  # Small delay between messages
        
        if random.random() < 0.1 and profile:
            asyncio.create_task(send_image_task(context, chat_id, arya_reply))
    
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text(get_random_error("general"))

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    
    logger.info(f"Voice message from user {user_id_telegram}")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error("voice"))
        return
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        transcribed_text = await asyncio.to_thread(transcribe_voice_sync, bytes(voice_bytes))
        if not transcribed_text:
            await update.message.reply_text(get_random_error("voice"))
            return
        
        save_to_memory(user_id, "user", f"[voice] {transcribed_text}")
        update_user_profile(user_id, {"last_message_from_user": datetime.now(timezone.utc).isoformat()})
        
        profile = get_user_profile(user_id)
        personality = profile.get("personality_choice", "female") if profile else "female"
        arya_reply = generate_response(user_id, transcribed_text, personality)
        
        if not arya_reply:
            await update.message.reply_text(get_random_error("general"))
            return
        
        save_to_memory(user_id, "arya", arya_reply)
        
        # Split into 2-3 messages if too long
        messages = split_into_messages(arya_reply)
        for msg in messages:
            await update.message.reply_text(msg)
            await asyncio.sleep(0.3)
    
    except Exception as e:
        logger.error(f"Error handling voice: {e}")
        await update.message.reply_text(get_random_error("voice"))

# ===== ASYNC TASK HANDLERS =====
async def send_image_task(context: CallbackContext, chat_id: int, prompt: str):
    """Generate and send image"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        photo = await asyncio.to_thread(generate_image_sync, prompt)
        
        if photo:
            await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="for you... 😉")
            logger.info("✅ Photo sent")
        else:
            await context.bot.send_message(chat_id=chat_id, text=get_random_error("image"))
    except Exception as e:
        logger.error(f"Error in send_image_task: {e}")

async def send_voice_task(context: CallbackContext, chat_id: int, text: str, user_id: str = None):
    """Generate and send voice"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
        
        voice_text = text[:500] if len(text) > 500 else text
        voice = await asyncio.to_thread(generate_voice_sync, voice_text)
        
        if voice:
            await context.bot.send_voice(chat_id=chat_id, voice=voice)
            logger.info("✅ Voice sent")
            if user_id:
                mark_voice_sent(user_id)
    except Exception as e:
        logger.error(f"Error in send_voice_task: {e}")


# =====================================================
# SECTION 5: BACKGROUND JOBS & MAIN
# =====================================================

async def check_for_checkins(context: CallbackContext):
    """Background job: Send 24-hour check-ins"""
    logger.info("Running check-in job")
    
    try:
        all_users = supabase.table("users").select("id, telegram_id").execute()
        
        for user in all_users.data:
            user_id = user["id"]
            telegram_id = user["telegram_id"]
            
            if should_send_checkin(user_id):
                checkins = [
                    "hey, what's up? 👀",
                    "missing you, what're you up to?",
                    "you there? 😊",
                    "just thinking about you",
                    "haven't heard from you, you good?",
                ]
                
                try:
                    await context.bot.send_message(chat_id=telegram_id, text=random.choice(checkins))
                    mark_checkin_sent(user_id)
                    logger.info(f"Check-in sent to {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send check-in: {e}")
    
    except Exception as e:
        logger.error(f"Error in check-in job: {e}")

def main():
    """Initialize and start the bot"""
    logger.info("="*70)
    logger.info("🚀 ARYA 4.0 - PRODUCTION FINAL VERSION")
    logger.info("="*70)
    logger.info(f"Brain: {BRAIN_MODEL}")
    logger.info(f"Voice In: {TRANSCRIPTION_PROVIDER}")
    logger.info(f"Voice Out: {TTS_PROVIDER}")
    logger.info(f"Images: {IMAGE_PROVIDER}")
    logger.info("="*70)
    logger.info("Features: Text, Voice, Images, Memory, Personalities, Check-ins")
    logger.info("="*70)
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    app.job_queue.run_repeating(check_for_checkins, interval=7200, first=10)
    
    logger.info("✅ All handlers registered!")
    logger.info("💬 BOT IS RUNNING - READY FOR MESSAGES")
    logger.info("="*70)
    
    app.run_polling()

if __name__ == '__main__':
    main()