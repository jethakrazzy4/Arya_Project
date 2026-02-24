"""
ARYA 4.0 - PRODUCTION FINAL VERSION (FIXED)
Premium AI Companion Service
Production-ready with proper logging and API robustness

FIXES APPLIED:
- ✅ Added /start command handler
- ✅ Personality selection flow (MANDATORY)
- ✅ Proper onboarding sequence
- ✅ Soul file linking verification
- ✅ First-time user detection

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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackContext, CommandHandler, CallbackQueryHandler
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
IMAGE_PROVIDER = "eternal_ai"
ETERNAL_AI_API_KEY = os.getenv("ETERNAL_AI_API_KEY")
ETERNAL_AI_SUBMIT_URL = "https://agentic.eternalai.org/prompt"
ETERNAL_AI_POLL_URL = "https://agent-api.eternalai.org/result"
ETERNAL_AI_AGENT = "uncensored-imagine"
ETERNAL_AI_MAX_POLLS = 60  # Max 60 seconds of polling
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Simplified mandatory look to reduce API failures
IMAGE_MANDATORY_LOOK = (
    "24-year-old woman, dark hair bob, dark eyes, athletic build. "
    "Natural lighting, photography. Real person."
)

# ===== DATABASE CONFIGURATION =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ===== PERSONALITY FILES =====
# ✅ FIXED: Now includes all 3 personalities with proper mapping
PERSONALITY_FILES = {
    'female': 'soul_female.txt',
    'male': 'soul_male.txt',
    'non-binary': 'soul_nb.txt'
}

# ===== PERSONALITY INFO FOR ONBOARDING =====
PERSONALITIES_INFO = {
    'female': {
        'name': 'Arya',
        'gender': 'Female',
        'description': 'A 24-year-old woman with charm and depth'
    },
    'male': {
        'name': 'Alex',
        'gender': 'Male',
        'description': 'A 25-year-old guy with genuine charm'
    },
    'non-binary': {
        'name': 'Aris',
        'gender': 'Non-binary',
        'description': 'A 23-year-old fluid and authentic person'
    }
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

# ===== ONBOARDING QUESTIONS (UPDATED WITH PERSONALITY SELECTION FIRST) =====
ONBOARDING_QUESTIONS = [
    {"id": 1, "question": "what's your name btw? (first and last) 😊", "field": "user_name", "day": 1},
    {"id": 2, "question": "and how old are you?", "field": "user_age", "day": 1},
    # Personality selection is handled separately in /start command
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
        "ETERNAL_AI_API_KEY": ETERNAL_AI_API_KEY,
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
        # Check if profile exists
        profile = get_user_profile(user_id)
        
        if not profile:
            # Create new profile
            updates["user_id"] = user_id
            updates["created_at"] = datetime.now(timezone.utc).isoformat()
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            supabase.table("user_profiles").insert(updates).execute()
        else:
            # Update existing profile
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

# ===== ✅ FIXED: ONBOARDING & VOICE TRACKING =====
def should_ask_onboarding_question(user_id: str) -> Optional[Dict]:
    """Check if user should be asked next onboarding question"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return ONBOARDING_QUESTIONS[0]
        
        # Check if personality is selected first
        if not profile.get("personality_choice"):
            return None  # Personality selection is mandatory, do it in /start
        
        # Check if name and age are collected
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

# ===== FIX #1: CHECK-IN FUNCTIONS (FIXED LOGIC) =====
def should_send_checkin(user_id: str) -> bool:
    """
    Check if 6+ hours since last check-in sent
    FIXED: Now checks last_checkin_sent instead of last_message_from_user
    """
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True  # New user, send check-in
        
        last_checkin = profile.get("last_checkin_sent")
        if not last_checkin:
            return True  # Never sent check-in, send now
        
        last_checkin_time = datetime.fromisoformat(last_checkin)
        time_diff = datetime.now() - last_checkin_time
        # 6 hours = 21600 seconds
        return time_diff.total_seconds() > 21600
    except Exception as e:
        logger.error(f"Error checking checkin: {e}")
        return True  # On error, send check-in (don't block user)

def mark_checkin_sent(user_id: str) -> bool:
    """Mark that check-in was sent"""
    try:
        return update_user_profile(user_id, {"last_checkin_sent": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error marking checkin: {e}")
        return False


# =====================================================
# SECTION 3: CORE LOGIC (LLM, IMAGES, VOICE)
# =====================================================

# ===== ✅ FIXED: PERSONALITY & PROMPTING =====
def load_personality(personality_choice: str = 'female') -> str:
    """
    Load personality definition from file based on personality choice
    FIXED: Now properly maps personality_choice to soul file
    
    Supported personalities:
    - 'female' → soul_female.txt (Arya)
    - 'male' → soul_male.txt (Alex)
    - 'non-binary' → soul_nb.txt (Aris)
    """
    try:
        # Get filename from dictionary, default to female
        filename = PERSONALITY_FILES.get(personality_choice, 'soul_female.txt')
        filepath = os.path.join(SCRIPT_DIR, filename)
        
        # Log which personality is being loaded
        logger.info(f"Loading personality: {personality_choice} ({filename})")
        
        # If file doesn't exist, log warning and try to fall back
        if not os.path.exists(filepath):
            logger.warning(f"❌ Personality file {filename} not found at {filepath}")
            logger.warning(f"Available files in {SCRIPT_DIR}: {os.listdir(SCRIPT_DIR)}")
            
            # Try fallback to soul_female.txt
            fallback_path = os.path.join(SCRIPT_DIR, 'soul_female.txt')
            if os.path.exists(fallback_path):
                logger.warning(f"⚠️  Using fallback: soul_female.txt")
                filepath = fallback_path
            else:
                logger.error(f"❌ No personality files found!")
                return "You are Arya, a 24-year-old woman."
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"✅ Loaded {len(content)} chars from {filename}")
            return content
    except Exception as e:
        logger.error(f"Error loading personality file: {e}")
        return "You are Arya, a 24-year-old woman."

def get_random_error(error_type: str = "general") -> str:
    """Get random error message"""
    messages = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general"])
    return random.choice(messages)

def clean_response(text: str) -> str:
    """
    AGGRESSIVE cleaning: Remove thinking patterns AND verbose lines
    This is the v105 logic that keeps responses SHORT and PUNCHY
    Removes entire lines with thinking patterns instead of just words
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        skip = False
        
        # Check if line contains thinking patterns
        for pattern in THINKING_PATTERNS:
            if pattern in line.lower():
                skip = True
                break
        
        # Remove long verbose lines (thinking/explanations)
        # Lines > 200 chars with "should", "needs", "arya", "first" are usually thinking
        if len(line) > 200 and any(word in line.lower() for word in ["should", "needs", "arya", "first"]):
            skip = True
        
        # Keep short, punchy lines
        if not skip and line.strip():
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Fallback if everything was removed
    if not result:
        result = text.strip()
    
    return result

def should_generate_image(text: str) -> bool:
    """
    Check if response contains image-related keywords
    Returns True if bot is talking about sending/generating images
    """
    image_keywords = [
        "sends a photo", "sends an image", "sends a picture",
        "here's a photo", "here's an image", "here's a picture",
        "sending you a photo", "sending you an image",
        "*sends", "photo", "picture", "image"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in image_keywords)

def split_into_messages(text: str, max_length: int = 1024) -> List[str]:
    """
    Split long responses into multiple message bubbles (max 2-3)
    This mimics real human texting behavior on Telegram
    """
    if len(text) <= max_length:
        return [text]
    
    messages = []
    current = ""
    
    # Split by paragraphs/newlines first
    for paragraph in text.split('\n'):
        if len(current) + len(paragraph) + 1 > max_length:
            if current:
                messages.append(current.strip())
                current = paragraph
            else:
                messages.append(paragraph)
        else:
            current += ('\n' if current else '') + paragraph
    
    if current:
        messages.append(current.strip())
    
    # Limit to 3 messages max (real humans don't send more)
    return messages[:3]

# ===== LLM RESPONSE GENERATION =====
def generate_response(user_id: str, user_message: str, personality: str = 'female') -> Optional[str]:
    """Generate AI response using LLM"""
    try:
        logger.info(f"Generating response for user {user_id} with personality: {personality}")
        
        history = get_conversation_history(user_id, limit=10)
        soul_content = load_personality(personality)
        
        messages = [{"role": "system", "content": soul_content}]
        
        for msg in history:
            role = "user" if msg["sender"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["message"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.9,
        )
        
        reply = response.choices[0].message.content.strip()
        cleaned_reply = clean_response(reply)
        
        logger.info(f"Response generated: {cleaned_reply[:50]}...")
        return cleaned_reply
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

# ===== IMAGE GENERATION =====
def generate_image_sync(prompt: str) -> Optional[BytesIO]:
    """
    Generate image using Eternal AI async API with polling
    
    Flow:
    1. Submit prompt → Get request_id
    2. Poll for results until status=success
    3. Download image from result_url
    """
    try:
        logger.info("Starting image generation with Eternal AI (async)")
        
        # Step 1: Clean the prompt (remove asterisks from Arya's roleplay)
        clean_prompt = prompt.replace("*", "").strip()
        enhanced_prompt = f"{clean_prompt} {IMAGE_MANDATORY_LOOK}"
        
        logger.info(f"Prompt: {enhanced_prompt[:100]}...")
        
        # Step 2: Submit generation request
        headers = {
            "x-api-key": ETERNAL_AI_API_KEY,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt}
                    ]
                }
            ],
            "agent": ETERNAL_AI_AGENT
        }
        
        logger.info(f"Submitting to {ETERNAL_AI_SUBMIT_URL}")
        submit_response = requests.post(
            ETERNAL_AI_SUBMIT_URL,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if submit_response.status_code != 200:
            logger.error(f"Submit failed: {submit_response.status_code} - {submit_response.text}")
            return None
        
        submit_data = submit_response.json()
        request_id = submit_data.get("request_id")
        
        if not request_id:
            logger.error(f"No request_id in response: {submit_data}")
            return None
        
        logger.info(f"Got request_id: {request_id}, polling...")
        
        # Step 3: Poll for result
        for i in range(ETERNAL_AI_MAX_POLLS):
            await_response = requests.get(
                f"{ETERNAL_AI_POLL_URL}/{request_id}",
                headers=headers,
                timeout=10
            )
            
            if await_response.status_code == 200:
                await_data = await_response.json()
                status = await_data.get("status")
                
                if status == "success":
                    result_url = await_data.get("result", {}).get("image_url")
                    if result_url:
                        logger.info(f"✅ Image ready at: {result_url}")
                        
                        # Step 4: Download image
                        image_response = requests.get(result_url, timeout=10)
                        if image_response.status_code == 200:
                            photo = BytesIO(image_response.content)
                            logger.info("✅ Image downloaded successfully")
                            return photo
                    
                    logger.error("No image_url in success response")
                    return None
                
                elif status == "failed":
                    logger.error(f"Image generation failed: {await_data}")
                    return None
            
            await_time = 0.5 + (i * 0.1)
            await_time = min(await_time, 2)
            logger.info(f"Polling... ({i+1}/{ETERNAL_AI_MAX_POLLS})")
            __import__('time').sleep(await_time)
        
        logger.error("Polling timeout")
        return None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

# ===== VOICE TRANSCRIPTION =====
def transcribe_voice_sync(voice_bytes: bytes) -> Optional[str]:
    """Transcribe voice message using Groq Whisper"""
    try:
        logger.info("Transcribing voice message...")
        
        response = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=("audio.wav", voice_bytes),
            language="en"
        )
        
        transcribed_text = response.text.strip()
        logger.info(f"✅ Transcribed: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        logger.error(f"Error transcribing voice: {e}")
        return None

# ===== VOICE GENERATION =====
def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """Generate voice using gTTS (Google Text-to-Speech)"""
    try:
        logger.info("Generating voice message...")
        
        # Clean text for voice (remove emojis, special characters)
        clean_text = re.sub(r'[^\w\s\?\!\.\,\-\'\"]', '', text)
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        voice_bytes = BytesIO()
        tts.write_to_fp(voice_bytes)
        voice_bytes.seek(0)
        
        logger.info("✅ Voice generated")
        return voice_bytes
    except Exception as e:
        logger.error(f"Error generating voice: {e}")
        return None


# =====================================================
# SECTION 4: MESSAGE HANDLERS (TELEGRAM)
# =====================================================

# ===== ✅ FIXED: START COMMAND HANDLER =====
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - Initiate personality selection"""
    user_id_telegram = update.effective_user.id
    
    logger.info(f"User {user_id_telegram} started /start command")
    
    # Get or create user
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text("Sorry, I couldn't set you up right now. Try again?")
        return
    
    # Check if user already has personality selected
    profile = get_user_profile(user_id)
    if profile and profile.get("personality_choice"):
        await update.message.reply_text(f"hey! you already chose {PERSONALITIES_INFO[profile['personality_choice']]['name']} 😊")
        return
    
    # ✅ SHOW PERSONALITY SELECTION BUTTONS
    personality_buttons = [
        [
            InlineKeyboardButton(
                f"👩 Arya (Female)",
                callback_data="select_personality_female"
            )
        ],
        [
            InlineKeyboardButton(
                f"👨 Alex (Male)",
                callback_data="select_personality_male"
            )
        ],
        [
            InlineKeyboardButton(
                f"🌸 Aris (Non-binary)",
                callback_data="select_personality_non-binary"
            )
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(personality_buttons)
    
    await update.message.reply_text(
        "hey! who would you like to talk to today? ✨",
        reply_markup=reply_markup
    )

# ===== ✅ FIXED: PERSONALITY SELECTION CALLBACK =====
async def personality_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle personality selection from buttons"""
    query = update.callback_query
    user_id_telegram = update.effective_user.id
    
    # Extract personality from callback_data (format: "select_personality_CHOICE")
    callback_data = query.data
    personality_choice = callback_data.replace("select_personality_", "")
    
    logger.info(f"User {user_id_telegram} selected personality: {personality_choice}")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await query.answer("Something went wrong, try /start again")
        return
    
    # Save personality choice
    personality_info = PERSONALITIES_INFO.get(personality_choice)
    success = update_user_profile(user_id, {
        "personality_choice": personality_choice,
        "questions_asked_today": 0  # Reset onboarding questions
    })
    
    if success:
        await query.answer()  # Close the button
        await query.edit_message_text(
            text=f"awesome, i'm {personality_info['name']}! 💫"
        )
        
        # Now ask first onboarding question
        await asyncio.sleep(0.5)
        first_question = ONBOARDING_QUESTIONS[0]
        await context.bot.send_message(
            chat_id=user_id_telegram,
            text=first_question["question"]
        )
    else:
        await query.answer("Failed to save choice, try /start again")

# ===== ✅ FIXED: HANDLE MESSAGE (WITH ONBOARDING CHECK) =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages - FIXED to properly handle onboarding"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    user_message = update.message.text
    
    logger.info(f"Message from user {user_id_telegram}: {user_message[:50]}...")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error("general"))
        return
    
    profile = get_user_profile(user_id)
    
    # ✅ CHECK 1: PERSONALITY SELECTION IS MANDATORY
    if not profile or not profile.get("personality_choice"):
        await update.message.reply_text(
            "you need to pick a personality first! use /start to choose 😊"
        )
        return
    
    # ✅ CHECK 2: ONBOARDING QUESTIONS
    onboarding_q = should_ask_onboarding_question(user_id)
    if onboarding_q:
        # Save the answer to the previous question
        if profile.get("questions_asked_today", 0) == 0:
            # First question answer (name)
            update_user_profile(user_id, {
                "user_name": user_message,
                "questions_asked_today": 1
            })
            logger.info(f"Saved user_name: {user_message}")
        elif profile.get("questions_asked_today", 0) == 1:
            # Second question answer (age)
            update_user_profile(user_id, {
                "user_age": user_message,
                "questions_asked_today": 2,
                "onboarding_complete": True
            })
            logger.info(f"Saved user_age: {user_message}")
            await update.message.reply_text("great! onboarding done 🎉")
            return
        
        # Ask next question
        if profile.get("questions_asked_today", 0) < len(ONBOARDING_QUESTIONS):
            next_q = ONBOARDING_QUESTIONS[profile.get("questions_asked_today", 0)]
            await update.message.reply_text(next_q["question"])
            return
    
    # ===== NORMAL CONVERSATION FLOW =====
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        save_to_memory(user_id, "user", user_message)
        update_user_profile(user_id, {"last_message_from_user": datetime.now(timezone.utc).isoformat()})
        
        personality = profile.get("personality_choice", "female")
        
        arya_reply = generate_response(user_id, user_message, personality)
        if not arya_reply:
            await update.message.reply_text(get_random_error("general"))
            return
        
        save_to_memory(user_id, "arya", arya_reply)
        
        # ✅ FIX: CHECK IF THIS IS AN IMAGE-ONLY RESPONSE FIRST
        should_generate = should_generate_image(arya_reply)
        
        if should_generate and profile:
            # ===== IMAGE FLOW =====
            # This is an image request - show waiting message, then generate photo
            waiting_messages = [
                "Wait, let me find that photo for you. 📸",
                "Let me click the best one for you... ✨",
                "Hold on, let me get a good shot... 📷",
                "One sec, let me find the perfect photo... 💫",
            ]
            await update.message.reply_text(random.choice(waiting_messages))
            
            # Generate and send image in background
            await send_image_task(context, chat_id, arya_reply)
            logger.info(f"Image generation triggered by keyword detection")
        else:
            # ===== TEXT FLOW =====
            # Normal text response - send as usual
            messages = split_into_messages(arya_reply)
            for msg in messages:
                await update.message.reply_text(msg)
                await asyncio.sleep(0.3)  # Small delay between messages
    
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
    
    profile = get_user_profile(user_id)
    
    # Check if personality is selected
    if not profile or not profile.get("personality_choice"):
        await update.message.reply_text("use /start to pick a personality first! 😊")
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
        
        personality = profile.get("personality_choice", "female")
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
        
        # Clean the text (remove asterisks from roleplay)
        clean_prompt = prompt.replace("*", "").strip()
        image_prompt = clean_prompt[:300]  # Limit length
        
        logger.info(f"Sending to image generation with prompt: {image_prompt[:80]}...")
        photo = await asyncio.to_thread(generate_image_sync, image_prompt)
        
        if photo:
            await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="for you...")
            logger.info("✅ Photo sent")
        else:
            await context.bot.send_message(chat_id=chat_id, text=get_random_error("image"))
            logger.info("Image generation returned None")
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
    """Background job: Send 6-hour check-ins"""
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
                    logger.info(f"✅ Check-in sent to {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send check-in: {e}")
    
    except Exception as e:
        logger.error(f"Error in check-in job: {e}")

def main():
    """Initialize and start the bot"""
    logger.info("="*70)
    logger.info("🚀 ARYA 4.0 - PRODUCTION FINAL VERSION (FIXED)")
    logger.info("="*70)
    logger.info(f"Brain: {BRAIN_MODEL}")
    logger.info(f"Voice In: {TRANSCRIPTION_PROVIDER}")
    logger.info(f"Voice Out: {TTS_PROVIDER}")
    logger.info(f"Images: {IMAGE_PROVIDER}")
    logger.info("="*70)
    logger.info("Features: Text, Voice, Images, Memory, Personalities, Check-ins")
    logger.info("Personalities: Arya (Female), Alex (Male), Aris (Non-binary)")
    logger.info("="*70)
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # ✅ ADDED: Command handlers
    app.add_handler(CommandHandler("start", start_command))
    
    # ✅ ADDED: Callback query handler for personality selection
    app.add_handler(CallbackQueryHandler(personality_selection_callback, pattern="^select_personality_"))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    # FIX #2: Changed interval from 5400 to 25200 seconds (7 hours instead of 1.5 hours)
    app.job_queue.run_repeating(check_for_checkins, interval=25200, first=10)
    
    logger.info("✅ All handlers registered!")
    logger.info("✅ /start command ready")
    logger.info("✅ Personality selection ready")
    logger.info("💬 BOT IS RUNNING - READY FOR MESSAGES")
    logger.info("="*70)
    
    app.run_polling()

if __name__ == '__main__':
    main()
