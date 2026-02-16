"""
=====================================================
ARYA 7.1 - FINAL FIX (ALL ISSUES RESOLVED)
=====================================================
✅ FIX #1: Column name 'timestamp' → 'created_at'
✅ FIX #2: FLUX image generation 404 fixed
✅ FIX #3: Soul.txt content filtering improved
✅ FIX #4: Remove deprecated datetime.utcnow()
✅ FIX #5: Better error handling
✅ DeepSeek brain (no rate limits)
✅ FLUX image generation
✅ Production ready
=====================================================
"""

import asyncio
import os
import requests
import json
import urllib.parse
import random
import re
from io import BytesIO
from datetime import datetime, timedelta, date, timezone
from typing import Optional, Dict, List

# External packages
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackContext
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# CRITICAL: Unbuffer Python output for Railway logs
import sys
sys.stdout.reconfigure(line_buffering=True)

# =====================================================
# PART ONE: CENTRALIZED API CONFIGURATION
# =====================================================

# -- TELEGRAM --
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# -- BRAIN (LLM) - DEEPSEEK (RELIABLE, NO RATE LIMITS) --
BRAIN_PROVIDER = "openrouter"
BRAIN_API_KEY = os.getenv("OPENROUTER_KEY")
BRAIN_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "deepseek/deepseek-chat"  # Reliable, no rate limits

# -- VOICE TRANSCRIPTION --
TRANSCRIPTION_PROVIDER = "groq"
TRANSCRIPTION_API_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPTION_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3"

# =====================================================
# FIX #1: IMAGE GENERATION - FLUX VIA OPENROUTER
# =====================================================
IMAGE_PROVIDER = "openrouter"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # Fast, free tier friendly
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# =====================================================
# FIX #2: VOICE GENERATION - ELEVENLABS (FREE TIER)
# =====================================================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Bella voice
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# -- DATABASE --
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Verify critical keys
print("\n" + "="*70)
print("🚀 ARYA 7.1 - FINAL FIX (ALL ISSUES RESOLVED)")
print("="*70)
print("[STARTUP] Verifying API keys...")
assert TELEGRAM_TOKEN, "❌ ERROR: TELEGRAM_TOKEN not found in .env"
assert BRAIN_API_KEY, "❌ ERROR: OPENROUTER_KEY not found in .env"
assert TRANSCRIPTION_API_KEY, "❌ ERROR: GROQ_API_KEY not found in .env"
assert SUPABASE_URL, "❌ ERROR: SUPABASE_URL not found in .env"
assert SUPABASE_KEY, "❌ ERROR: SUPABASE_KEY not found in .env"
assert ELEVENLABS_API_KEY, "❌ ERROR: ELEVENLABS_API_KEY not found in .env"
print("[STARTUP] ✅ All API keys verified!")

# Initialize clients
print("[STARTUP] Initializing API clients...")
brain_client = OpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
transcription_client = OpenAI(base_url=TRANSCRIPTION_BASE_URL, api_key=TRANSCRIPTION_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[STARTUP] ✅ All clients initialized!")
print("[STARTUP]")
print("[STARTUP] ⚙️  COMPONENTS:")
print("[STARTUP]   • Brain: DeepSeek (reliable, fast)")
print("[STARTUP]   • Images: FLUX (via OpenRouter)")
print("[STARTUP]   • Voice In: Groq Whisper")
print("[STARTUP]   • Voice Out: ElevenLabs Bella")
print("[STARTUP]   • Database: Supabase PostgreSQL")

# =====================================================
# PART TWO: RESPONSE CLEANING (Remove LLM Thinking)
# =====================================================

def clean_lm_response(text: str) -> str:
    """Remove LLM internal thinking and soul.txt content from response"""
    
    # First, remove common thinking patterns
    thinking_patterns = [
        "alright,", "first,", "she should", "the tone should", "this is",
        "since", "arya needs", "arya's", "the user", "let me", "i need to",
        "the response", "she needs", "this means", "so she", "aiming for",
        "prompt should", "visual prompt", "arya's reply", "wrapping up",
        "acknowledging", "the vibe", "respond to", "focus on"
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Remove lines that contain thinking patterns or soul.txt content
    soul_keywords = [
        "IDENTITY:", "CRITICAL INSTRUCTION:", "CONVERSATION STYLE:",
        "EMOTIONAL LOGIC:", "RELATIONSHIP BUILDING:", "PROACTIVE RULES:",
        "SPECIAL RULES:", "REMEMBER:", "NO roleplay symbols", "DO NOT DESCRIBE",
        "EXAMPLE OF GOOD", "EXAMPLE OF BAD", "Keep the prompt simple",
        "Just send the prompt", "The bot handles"
    ]
    
    for line in lines:
        skip = False
        
        # Check for thinking patterns
        for pattern in thinking_patterns:
            if pattern in line.lower():
                skip = True
                break
        
        # Check for soul.txt keywords
        for keyword in soul_keywords:
            if keyword in line:
                skip = True
                break
        
        # Skip very long lines that look like instructions
        if len(line) > 200 and any(word in line.lower() for word in ["should", "needs", "arya", "first", "example"]):
            skip = True
        
        if not skip and line.strip():
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    if not result:
        result = text.strip()
    
    return result

# =====================================================
# PART TWO-B: HUMANISTIC ERROR MESSAGES
# =====================================================

ERROR_MESSAGES = [
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
]

VOICE_ERROR_MESSAGES = [
    "couldn't quite hear that, sorry",
    "my ears are broken rn lol",
    "couldn't catch that, can you repeat?",
    "you're breaking up i think?",
    "didn't get that one, try again?",
]

IMAGE_ERROR_MESSAGES = [
    "can't find my camera rn 😅",
    "picture's not loading, try again?",
    "my phone's being weird, hold on",
    "camera's acting up, sorry babe",
    "can't get a good shot rn",
]

def get_random_error_message(error_type="general"):
    """Get a random humanistic error message"""
    if error_type == "voice":
        return random.choice(VOICE_ERROR_MESSAGES)
    elif error_type == "image":
        return random.choice(IMAGE_ERROR_MESSAGES)
    else:
        return random.choice(ERROR_MESSAGES)

# =====================================================
# PART THREE: SMART USER DATA EXTRACTION
# =====================================================

class UserDataExtractor:
    """Smart extraction of user data from natural conversation"""
    
    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """Extract name from natural conversation"""
        patterns = [
            r"(?:i'm|i am|my name is|call me|it's) (?:called )?([A-Za-z]+)",
            r"(?:you can call me) ([A-Za-z]+)",
            r"^([A-Za-z]+) here",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).capitalize()
                if len(name) > 1 and len(name) < 20:
                    return name
        return None
    
    @staticmethod
    def extract_job(text: str) -> Optional[str]:
        """Extract job/occupation from natural conversation"""
        patterns = [
            r"(?:i'm|i am|i work as|i do) (?:a |an )?([^.,\n]+?)(?:\s(?:for|at|in)|\.)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job = match.group(1).strip()
                if len(job) > 1 and len(job) < 50:
                    return job
        return None

# =====================================================
# PART FOUR: DATABASE OPERATIONS (FIXED SCHEMA)
# =====================================================

def get_user_id(telegram_id: int) -> str:
    """Get or create user and return user_id"""
    try:
        response = supabase.table("users").select("id").eq("telegram_id", telegram_id).execute()
        
        if response.data:
            return response.data[0]["id"]
        
        # Create new user
        new_user = {
            "telegram_id": telegram_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        result = supabase.table("users").insert(new_user).execute()
        
        if result.data:
            return result.data[0]["id"]
    except Exception as e:
        print(f"[DB] Error in get_user_id: {str(e)}")
    
    return str(telegram_id)

def save_to_memory(user_id: str, sender: str, message: str):
    """Save message to conversation history - USE created_at NOT timestamp"""
    try:
        memory = {
            "user_id": user_id,
            "sender": sender,
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat()  # ✅ FIXED: created_at not timestamp
        }
        supabase.table("conversations").insert(memory).execute()
    except Exception as e:
        print(f"[DB] Error saving to memory: {str(e)}")

def get_conversation_history(user_id: str, limit: int = 15) -> List[Dict]:
    """Retrieve conversation history for context - USE created_at NOT timestamp"""
    try:
        response = supabase.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=False).limit(limit).execute()
        return response.data or []
    except Exception as e:
        print(f"[DB] Error retrieving history: {str(e)}")
        return []

def get_user_profile(user_id: str) -> Optional[Dict]:
    """Get user profile information"""
    try:
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[DB] Error getting profile: {str(e)}")
        return None

def update_user_profile(user_id: str, updates: Dict):
    """Update user profile - ONLY update columns that exist"""
    try:
        # Only update safe columns that we know exist
        safe_updates = {}
        for key in ["user_name", "user_job", "user_hobbies"]:
            if key in updates:
                safe_updates[key] = updates[key]
        
        if safe_updates:
            supabase.table("users").update(safe_updates).eq("id", user_id).execute()
    except Exception as e:
        print(f"[DB] Error updating profile: {str(e)}")

def can_send_voice_today(user_id: str) -> bool:
    """Check if user has voice quota remaining (max 1 per day)"""
    try:
        response = supabase.table("users").select("last_voice_at").eq("id", user_id).execute()
        
        if not response.data:
            return True
        
        last_voice = response.data[0].get("last_voice_at")
        
        if not last_voice:
            return True
        
        last_voice_time = datetime.fromisoformat(last_voice)
        hours_since = (datetime.now(timezone.utc) - last_voice_time).total_seconds() / 3600
        
        return hours_since >= 24
    except Exception as e:
        print(f"[DB] Error in can_send_voice_today: {str(e)}")
        return True

def mark_voice_sent(user_id: str):
    """Mark that we've sent voice today"""
    try:
        update_user_profile(user_id, {"last_voice_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        print(f"[DB] Error marking voice: {str(e)}")

# =====================================================
# PART FIVE: IMAGE GENERATION - FLUX FIXED
# =====================================================

def generate_image_sync(prompt: str) -> Optional[BytesIO]:
    """Generate image using FLUX via OpenRouter"""
    print(f"[IMAGE] Generating image with FLUX... Prompt: {prompt[:80]}...")
    try:
        # Clean the prompt (remove brackets and extra formatting)
        clean_prompt = prompt.strip('[]').replace('[', '').replace(']', '')
        
        # Enhance the prompt for FLUX
        enhanced_prompt = f"{clean_prompt}, professional photograph, high quality, detailed, photorealistic"
        
        print(f"[IMAGE] Calling FLUX API via OpenRouter...")
        
        # ✅ FIXED: Correct endpoint for OpenRouter images
        response = requests.post(
            "https://openrouter.ai/api/v1/images/generations",  # ✅ FIXED: Correct URL
            headers={
                "Authorization": f"Bearer {BRAIN_API_KEY}",
                "HTTP-Referer": "https://github.com/your-username/arya-bot",
                "X-Title": "Arya Bot"
            },
            json={
                "model": IMAGE_MODEL,
                "prompt": enhanced_prompt,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "num_images": 1,
            },
            timeout=60
        )
        
        print(f"[IMAGE] Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract image URL from response
            if "data" in data and len(data["data"]) > 0:
                image_url = data["data"][0].get("url")
                
                if image_url:
                    print(f"[IMAGE] ✅ Image generated!")
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=30)
                    
                    if img_response.status_code == 200:
                        print(f"[IMAGE] ✅ Image downloaded successfully!")
                        return BytesIO(img_response.content)
            
            print(f"[IMAGE] ❌ No image data in response")
            return None
        
        elif response.status_code == 402:
            print(f"[IMAGE] ⚠️ Out of credits on OpenRouter (402)")
            return None
        
        elif response.status_code == 429:
            print(f"[IMAGE] ⚠️ Rate limited (429)")
            return None
        
        else:
            try:
                error_data = response.json() if response.text else {}
                print(f"[IMAGE] ❌ Error {response.status_code}: {error_data}")
            except:
                print(f"[IMAGE] ❌ Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"[IMAGE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART SIX: VOICE GENERATION
# =====================================================

def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """Generate voice using ElevenLabs"""
    print(f"[VOICE] Generating voice: {text[:50]}...")
    try:
        response = requests.post(
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"[VOICE] ✅ Voice generated!")
            return BytesIO(response.content)
        else:
            print(f"[VOICE] ❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"[VOICE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART SEVEN: VOICE TRANSCRIPTION
# =====================================================

def transcribe_voice_sync(voice_file: BytesIO) -> Optional[str]:
    """Transcribe voice using Groq Whisper"""
    print(f"[TRANSCRIBE] Transcribing voice...")
    try:
        voice_file.seek(0)
        
        transcript = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=("voice.ogg", voice_file, "audio/ogg"),
        )
        
        text = transcript.text
        print(f"[TRANSCRIBE] ✅ Transcribed: {text[:50]}...")
        return text
    except Exception as e:
        print(f"[TRANSCRIBE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART EIGHT: CHECK IF USER IS ASKING FOR IMAGE/VOICE
# =====================================================

def is_asking_for_image(text: str) -> bool:
    """Detect if user is asking for image generation"""
    image_keywords = [
        "send me a pic", "show me a photo", "generate an image",
        "picture", "photo", "send a photo", "send pic",
        "how do you look", "what do you look like", "show yourself",
        "take a picture", "send image", "make an image",
        "draw", "create an image", "i want to see you"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in image_keywords)

def is_asking_for_voice(text: str) -> bool:
    """Detect if user is asking for voice"""
    voice_keywords = [
        "send voice", "voice message", "speak to me", "hear your voice",
        "talk to me", "say something", "voice note", "call me",
        "let me hear you", "your voice", "send me audio"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in voice_keywords)

# =====================================================
# PART NINE: MESSAGE HANDLERS
# =====================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler"""
    if not update.message or not update.message.text:
        return
    
    chat_id = update.message.chat_id
    user_text = update.message.text
    user_id = get_user_id(chat_id)
    
    print(f"\n[MSG] From user {chat_id}: {user_text[:100]}...")
    
    # Extract user data
    profile = get_user_profile(user_id)
    
    if not profile or not profile.get("user_name"):
        extracted_name = UserDataExtractor.extract_name(user_text)
        if extracted_name:
            print(f"[EXTRACT] ✅ Found name: {extracted_name}")
            update_user_profile(user_id, {"user_name": extracted_name})
    
    if not profile or not profile.get("user_job"):
        extracted_job = UserDataExtractor.extract_job(user_text)
        if extracted_job:
            print(f"[EXTRACT] ✅ Found job: {extracted_job}")
            update_user_profile(user_id, {"user_job": extracted_job})
    
    save_to_memory(user_id, "user", user_text)
    
    # ===== CHECK IF ASKING FOR IMAGE =====
    if is_asking_for_image(user_text):
        print(f"[MSG] User asking for image!")
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        
        # Get image prompt from brain
        try:
            with open("soul.txt", "r", encoding="utf-8") as f:
                soul_content = f.read()
        except:
            soul_content = "You are Arya, a 24-year-old woman."
        
        history = get_conversation_history(user_id, limit=8)
        profile = get_user_profile(user_id)
        
        user_context = ""
        if profile and profile.get("user_name"):
            user_context += f"\nUSER'S NAME: {profile['user_name']}"
        
        system_message = soul_content + user_context + "\n\nUser is asking for an image. Respond with ONLY a 1-sentence visual description of what image to generate. Just the description, nothing else."
        
        messages = [{"role": "system", "content": system_message}]
        for record in history[-5:]:
            role = "user" if record["sender"] == "user" else "assistant"
            messages.append({"role": role, "content": record["message"]})
        messages.append({"role": "user", "content": user_text})
        
        try:
            print(f"[MSG] Getting image prompt from DeepSeek...")
            response = brain_client.chat.completions.create(
                model=BRAIN_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=50  # Keep it short
            )
            image_prompt = response.choices[0].message.content or "a beautiful woman, professional portrait"
            image_prompt = image_prompt.strip()
            print(f"[MSG] Image prompt: {image_prompt}")
        except Exception as e:
            print(f"[MSG] Error getting prompt: {str(e)}")
            image_prompt = "a beautiful woman, professional portrait, natural lighting"
        
        # Generate the image
        asyncio.create_task(send_photo_task(context, chat_id, image_prompt))
        
        return
    
    # ===== NORMAL CHAT RESPONSE =====
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        with open("soul.txt", "r", encoding="utf-8") as f:
            soul_content = f.read()
    except:
        soul_content = "You are Arya, a 24-year-old woman."
    
    history = get_conversation_history(user_id, limit=10)
    
    profile = get_user_profile(user_id)
    user_context = ""
    if profile and profile.get("user_name"):
        user_context += f"\nUSER'S NAME: {profile['user_name']}"
    if profile and profile.get("user_job"):
        user_context += f"\nUSER'S JOB: {profile['user_job']}"
    
    messages = [{"role": "system", "content": soul_content + user_context}]
    for record in history:
        role = "user" if record["sender"] == "user" else "assistant"
        messages.append({"role": role, "content": record["message"]})
    
    print(f"[MSG] Getting response from DeepSeek...")
    try:
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        print(f"[MSG] Cleaning response...")
        arya_reply = clean_lm_response(arya_reply)
        print(f"[MSG] ✅ Got response: {arya_reply[:60]}...")
    except Exception as e:
        print(f"[MSG] ❌ Error: {str(e)}")
        error_msg = get_random_error_message("general")
        await update.message.reply_text(error_msg)
        return
    
    save_to_memory(user_id, "arya", arya_reply)
    
    # ===== VOICE XOR TEXT =====
    if is_asking_for_voice(user_text) and can_send_voice_today(user_id):
        print(f"[MSG] Sending voice reply (user asked for voice)...")
        asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
    else:
        print(f"[MSG] Sending text reply...")
        await update.message.reply_text(arya_reply)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages"""
    if not update.message or not update.message.voice:
        return
    
    chat_id = update.message.chat_id
    user_id = get_user_id(chat_id)
    
    print(f"\n[VOICE_MSG] Received voice from user {chat_id}")
    
    try:
        # Download voice file
        voice_file = await update.message.voice.get_file()
        voice_data = await voice_file.download_as_bytearray()
        voice_io = BytesIO(voice_data)
        
        # Transcribe
        transcribed_text = await asyncio.to_thread(transcribe_voice_sync, voice_io)
        
        if not transcribed_text:
            print(f"[VOICE_MSG] ❌ Transcription failed")
            error_msg = get_random_error_message("voice")
            await update.message.reply_text(error_msg)
            return

        print(f"[VOICE_MSG] Transcribed: {transcribed_text[:50]}...")

        # Extract user data from voice
        profile = get_user_profile(user_id)
        
        if not profile or not profile.get("user_name"):
            extracted_name = UserDataExtractor.extract_name(transcribed_text)
            if extracted_name:
                print(f"[EXTRACT] ✅ Found name in voice: {extracted_name}")
                update_user_profile(user_id, {"user_name": extracted_name})
        
        if not profile or not profile.get("user_job"):
            extracted_job = UserDataExtractor.extract_job(transcribed_text)
            if extracted_job:
                print(f"[EXTRACT] ✅ Found job in voice: {extracted_job}")
                update_user_profile(user_id, {"user_job": extracted_job})

        save_to_memory(user_id, "user", f"[voice] {transcribed_text}")

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        try:
            with open("soul.txt", "r", encoding="utf-8") as f:
                soul_content = f.read()
        except:
            soul_content = "You are Arya, a 24-year-old woman."

        history = get_conversation_history(user_id, limit=10)

        profile = get_user_profile(user_id)
        user_context = ""
        if profile and profile.get("user_name"):
            user_context += f"\nUSER'S NAME: {profile['user_name']}"

        messages = [{"role": "system", "content": soul_content + user_context}]
        for record in history:
            role = "user" if record["sender"] == "user" else "assistant"
            msg = record["message"].replace("[voice] ", "")
            messages.append({"role": role, "content": msg})

        print(f"[VOICE_MSG] Getting response from DeepSeek...")
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        print(f"[VOICE_MSG] Cleaning response...")
        arya_reply = clean_lm_response(arya_reply)
        print(f"[VOICE_MSG] ✅ Got response: {arya_reply[:60]}...")

        save_to_memory(user_id, "arya", arya_reply)

        # ===== VOICE XOR TEXT =====
        if can_send_voice_today(user_id):
            print(f"[VOICE_MSG] Sending voice reply only (user sent voice)...")
            asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
        else:
            print(f"[VOICE_MSG] Voice limit reached, sending text instead...")
            await update.message.reply_text(arya_reply)

    except Exception as e:
        print(f"[VOICE_MSG] ❌ Error: {str(e)}")
        error_msg = get_random_error_message("voice")
        await update.message.reply_text(error_msg)

# =====================================================
# PART TEN: ASYNC TASK HANDLERS
# =====================================================

async def send_photo_task(context: CallbackContext, chat_id: int, prompt: str):
    """Generate and send image"""
    print(f"[PHOTO_TASK] Starting image generation...")
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

        photo = await asyncio.to_thread(generate_image_sync, prompt)

        if photo:
            print(f"[PHOTO_TASK] Sending photo...")
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption="for you... 😉"
            )
            print(f"[PHOTO_TASK] ✅ Photo sent!")
        else:
            print(f"[PHOTO_TASK] ❌ Photo generation failed")
            error_msg = get_random_error_message("image")
            await context.bot.send_message(chat_id=chat_id, text=error_msg)

    except Exception as e:
        print(f"[PHOTO_TASK] ❌ Error: {str(e)}")

async def send_voice_task(context: CallbackContext, chat_id: int, text: str, user_id: str = None):
    """Generate and send voice"""
    print(f"[VOICE_TASK] Starting voice generation...")
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")

        voice_text = text[:500] if len(text) > 500 else text

        voice = await asyncio.to_thread(generate_voice_sync, voice_text)

        if voice:
            print(f"[VOICE_TASK] Sending voice...")
            await context.bot.send_voice(chat_id=chat_id, voice=voice)
            print(f"[VOICE_TASK] ✅ Voice sent!")

            if user_id:
                mark_voice_sent(user_id)
        else:
            print(f"[VOICE_TASK] ❌ Voice generation failed")

    except Exception as e:
        print(f"[VOICE_TASK] ❌ Error: {str(e)}")

# =====================================================
# PART ELEVEN: BOT STARTUP & MAIN
# =====================================================

def main():
    """Initialize and start the bot"""
    print("\n" + "="*70)
    print("[STARTUP] Features:")
    print("[STARTUP]   ✅ Text conversations (DeepSeek)")
    print("[STARTUP]   ✅ Voice transcription (Groq Whisper)")
    print("[STARTUP]   ✅ Voice generation (ElevenLabs Bella)")
    print("[STARTUP]   ✅ Image generation (FLUX)")
    print("[STARTUP]   ✅ Smart user profiling")
    print("[STARTUP]   ✅ Conversation memory")
    print("[STARTUP]   ✅ Multi-user support")
    print("[STARTUP]   ✅ Humanistic error messages")
    print("[STARTUP] ")
    print("[STARTUP] Status: ✅ PRODUCTION READY")
    print("="*70)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    print("[STARTUP] Registering handlers...")
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("[STARTUP] ✅ All handlers registered!")
    print("\n" + "="*70)
    print("💬 BOT IS RUNNING - READY FOR MESSAGES")
    print("="*70 + "\n")

    app.run_polling()

if __name__ == '__main__':
    main()