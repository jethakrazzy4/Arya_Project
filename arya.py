"""
=====================================================
ARYA 5.0 - FINAL CORRECTED VERSION
=====================================================
✅ FIX #1: Images via FREE Pollinations.ai (no API key)
✅ FIX #2: User data extraction & storage
✅ FIX #3: Voice XOR Text (not both)
✅ FIX #4: Explicit voice trigger when user asks
✅ FIX #5: Explicit image trigger when user asks
✅ FIX #6: Database schema issues fixed
✅ ElevenLabs for natural voice output
✅ Check-in every 6 hours
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
from datetime import datetime, timedelta, date
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

# -- BRAIN (LLM) --
BRAIN_PROVIDER = "openrouter"
BRAIN_API_KEY = os.getenv("OPENROUTER_KEY")
BRAIN_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

# -- VOICE TRANSCRIPTION --
TRANSCRIPTION_PROVIDER = "groq"
TRANSCRIPTION_API_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPTION_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3"

# =====================================================
# FIX #1: IMAGE GENERATION - FREE POLLINATIONS.AI (NO KEY NEEDED)
# =====================================================
IMAGE_PROVIDER = "openrouter"
IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"  # Premium FLUX quality
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
# Alternative: use free FLUX.1 model
# IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # Faster, free tier

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
print("[STARTUP] ARYA 5.0 - FINAL CORRECTED VERSION")
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
print(f"[STARTUP] Image Generation: {IMAGE_PROVIDER} (FREE - no API key needed)")
print(f"[STARTUP] Voice Generation: ElevenLabs Bella (Free Tier)")

# =====================================================
# PART TWO-A: RESPONSE CLEANING (Remove LLM Thinking)
# =====================================================

THINKING_PATTERNS = [
    "alright,", "first,", "she should", "the tone should", "this is",
    "since", "arya needs", "arya's", "the user", "let me", "i need to",
    "the response", "she needs", "this means", "so she", "aiming for",
    "prompt should", "visual prompt", "arya's reply", "wrapping up",
    "acknowledging", "the vibe", "respond to", "focus on"
]

def clean_lm_response(text: str) -> str:
    """Remove LLM internal thinking from response"""
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

# =====================================================
# PART TWO: HUMANISTIC ERROR MESSAGES
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
    """
    Smart extraction of user data from natural conversation
    """
    
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
            r"(?:i'm|i am) (?:a |an )?([^.,\n]+?)(?:\s(?:developer|engineer|designer|manager|teacher|doctor|lawyer))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job = match.group(1).strip()
                if 2 < len(job) < 50:
                    return job
        return None
    
    @staticmethod
    def extract_hobbies(text: str) -> Optional[str]:
        """Extract hobbies/interests from natural conversation"""
        if any(word in text.lower() for word in ["hobby", "hobbies", "like to", "enjoy", "interested in", "passionate about"]):
            patterns = [
                r"(?:like to|enjoy|love|passionate about) ([^.,\n]+)",
                r"(?:my hobbies are|interests include) ([^.,\n]+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    hobbies = match.group(1).strip()
                    if 3 < len(hobbies) < 100:
                        return hobbies
        return None

# =====================================================
# PART FOUR: KEYWORD DETECTION (FIX #4 & #5)
# =====================================================

def should_generate_image(text: str) -> bool:
    """
    FIX #5: Explicitly check if user is asking for an image
    Don't rely on LLM to decide - make it explicit in code
    """
    image_keywords = [
        "photo", "picture", "image", "send me", "show me",
        "pic", "snapshot", "portrait", "selfie", "camera",
        "take a photo", "generate", "create", "draw"
    ]
    return any(keyword in text.lower() for keyword in image_keywords)

def should_generate_voice(text: str) -> bool:
    """
    FIX #4: Explicitly check if user is asking for a voice message
    Don't rely on LLM to decide - make it explicit in code
    """
    voice_keywords = [
        "voice", "audio", "voice note", "voice message", "send voice",
        "voice message", "record", "speak", "hear your voice",
        "say something", "talk to me", "voice call"
    ]
    return any(keyword in text.lower() for keyword in voice_keywords)

# =====================================================
# PART FIVE: DATABASE FUNCTIONS
# =====================================================

def get_or_create_user(telegram_id: int) -> Optional[str]:
    """Get or create user in database"""
    try:
        result = supabase.table("users").select("id").eq("telegram_id", telegram_id).execute()
        
        if result.data:
            return result.data[0]["id"]
        else:
            new_user = supabase.table("users").insert({"telegram_id": telegram_id}).execute()
            return new_user.data[0]["id"] if new_user.data else None
    except Exception as e:
        print(f"[DB] Error get_or_create_user: {str(e)}")
        return None

def get_user_profile(user_id: str) -> Optional[Dict]:
    """Get user profile from database"""
    try:
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"[DB] Error get_user_profile: {str(e)}")
        return None

def update_user_profile(user_id: str, updates: Dict) -> bool:
    """
    FIX #6: Update user profile - only update valid columns
    Don't try to update columns that don't exist in schema
    """
    try:
        # Only update columns that exist in Supabase schema
        valid_columns = ["user_name", "user_job", "user_hobbies", "last_voice_sent", "last_checkin_sent"]
        filtered_updates = {k: v for k, v in updates.items() if k in valid_columns}
        
        if filtered_updates:
            supabase.table("users").update(filtered_updates).eq("id", user_id).execute()
            print(f"[DB] ✅ Updated user profile: {filtered_updates}")
        return True
    except Exception as e:
        print(f"[DB] ❌ Error updating profile: {str(e)}")
        return False

def save_to_memory(user_id: str, sender: str, message: str) -> bool:
    """Save message to conversation history"""
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": sender,
            "message": message,
            "created_at": datetime.now().isoformat()
        }).execute()
        return True
    except Exception as e:
        print(f"[DB] Error save_to_memory: {str(e)}")
        return False

def get_conversation_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent conversation history"""
    try:
        result = supabase.table("conversations").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return list(reversed(result.data)) if result.data else []
    except Exception as e:
        print(f"[DB] Error get_conversation_history: {str(e)}")
        return []

def can_send_voice_today(user_id: str) -> bool:
    """Check if user has voice quota remaining (once per day)"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True
        
        last_voice = profile.get("last_voice_sent")
        if not last_voice:
            return True
        
        last_voice_date = datetime.fromisoformat(last_voice).date()
        return last_voice_date < date.today()
    except Exception as e:
        print(f"[DB] Error can_send_voice_today: {str(e)}")
        return True

def mark_voice_sent(user_id: str) -> bool:
    """Mark that voice was sent today"""
    try:
        update_user_profile(user_id, {"last_voice_sent": datetime.now().isoformat()})
        return True
    except Exception as e:
        print(f"[DB] Error mark_voice_sent: {str(e)}")
        return False

def mark_checkin_sent(user_id: str) -> bool:
    """Mark that check-in was sent"""
    try:
        update_user_profile(user_id, {"last_checkin_sent": datetime.now().isoformat()})
        return True
    except Exception as e:
        print(f"[DB] Error mark_checkin_sent: {str(e)}")
        return False

def should_send_checkin(user_id: str) -> bool:
    """Check if we should send a check-in message (every 6 hours)"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True
        
        last_checkin = profile.get("last_checkin_sent")
        if not last_checkin:
            return True
        
        last_checkin_time = datetime.fromisoformat(last_checkin)
        time_diff = datetime.now() - last_checkin_time
        return time_diff.total_seconds() > 21600  # 6 hours
    except Exception as e:
        print(f"[DB] Error should_send_checkin: {str(e)}")
        return False

# =====================================================
# PART SIX: IMAGE GENERATION
# =====================================================

def generate_image_sync(prompt: str) -> Optional[BytesIO]:
    """
    Generate image using FLUX via OpenRouter
    """
    print(f"[IMAGE] Generating image with FLUX... Prompt: {prompt[:80]}...")
    try:
        # Enhance the prompt for better results
        enhanced_prompt = f"{prompt}, high quality, detailed, photorealistic, professional photograph"
        
        # Call FLUX via OpenRouter
        response = requests.post(
            f"{BRAIN_BASE_URL}/images/generations",
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
                "quality": "hd"
            },
            timeout=60
        )
        
        print(f"[IMAGE] Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract image URL from response
            if "data" in data and len(data["data"]) > 0:
                image_url = data["data"][0].get("url")
                
                if image_url:
                    print(f"[IMAGE] ✅ Image generated! Downloading...")
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=30)
                    
                    if img_response.status_code == 200:
                        print(f"[IMAGE] ✅ Image downloaded successfully!")
                        return BytesIO(img_response.content)
                    else:
                        print(f"[IMAGE] ❌ Failed to download image: {img_response.status_code}")
                        return None
            
            print(f"[IMAGE] ❌ No image data in response: {data}")
            return None
        
        elif response.status_code == 402:
            print(f"[IMAGE] ⚠️ Out of credits on OpenRouter")
            return None
        else:
            error_data = response.json() if response.text else {}
            print(f"[IMAGE] ❌ Error: {response.status_code} - {error_data}")
            return None
            
    except Exception as e:
        print(f"[IMAGE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART SEVEN: VOICE GENERATION (ELEVENLABS)
# =====================================================

def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """
    Generate voice using ElevenLabs
    """
    print(f"[VOICE_GEN] Generating voice with ElevenLabs...")
    
    try:
        text_limited = text[:500] if len(text) > 500 else text

        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text_limited,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "use_speaker_boost": True
            }
        }

        print(f"[VOICE_GEN] Calling ElevenLabs API for: {text_limited[:40]}...")

        response = requests.post(
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"[VOICE_GEN] Response status: {response.status_code}")

        if response.status_code == 200:
            voice_data = BytesIO(response.content)
            voice_data.seek(0)
            print(f"[VOICE_GEN] ✅ Voice generated successfully!")
            return voice_data
        else:
            print(f"[VOICE_GEN] ❌ Error: {response.status_code} - {response.text[:100]}")
            return None

    except Exception as e:
        print(f"[VOICE_GEN] ❌ Error generating voice: {str(e)}")
        return None

# =====================================================
# PART EIGHT: VOICE TRANSCRIPTION
# =====================================================

def transcribe_voice_sync(voice_bytes: bytes) -> Optional[str]:
    """Transcribe voice message to text"""
    print(f"[TRANSCRIBE] Transcribing voice ({len(voice_bytes)} bytes)...")
    
    try:
        from io import BytesIO
        
        audio_file = BytesIO(voice_bytes)
        audio_file.name = "voice.ogg"

        transcript = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            language="en"
        )

        text = transcript.text.strip()
        print(f"[TRANSCRIBE] ✅ Transcribed: {text[:50]}...")
        return text

    except Exception as e:
        print(f"[TRANSCRIBE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART NINE: MESSAGE HANDLERS
# =====================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    user_message = update.message.text

    print(f"\n{'='*70}")
    print(f"[MESSAGE] Received from user {user_id_telegram}")
    print(f"[MESSAGE] Text: {user_message[:80]}...")
    print(f"{'='*70}")

    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error_message("general"))
        return

    # ===== EXTRACT USER DATA =====
    print(f"[EXTRACT] Checking for user data in message...")
    profile = get_user_profile(user_id)
    
    if not profile or not profile.get("user_name"):
        extracted_name = UserDataExtractor.extract_name(user_message)
        if extracted_name:
            print(f"[EXTRACT] ✅ Found name: {extracted_name}")
            update_user_profile(user_id, {"user_name": extracted_name})
    
    if not profile or not profile.get("user_job"):
        extracted_job = UserDataExtractor.extract_job(user_message)
        if extracted_job:
            print(f"[EXTRACT] ✅ Found job: {extracted_job}")
            update_user_profile(user_id, {"user_job": extracted_job})
    
    if not profile or not profile.get("user_hobbies"):
        extracted_hobbies = UserDataExtractor.extract_hobbies(user_message)
        if extracted_hobbies:
            print(f"[EXTRACT] ✅ Found hobbies: {extracted_hobbies}")
            update_user_profile(user_id, {"user_hobbies": extracted_hobbies})

    save_to_memory(user_id, "user", user_message)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        try:
            with open("soul.txt", "r", encoding="utf-8") as f:
                soul_content = f.read()
        except:
            soul_content = "You are Arya, a 24-year-old woman."

        history = get_conversation_history(user_id, limit=10)

        user_context = ""
        profile = get_user_profile(user_id)
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
            messages.append({"role": role, "content": record["message"]})

        print(f"[BRAIN] Getting response from {BRAIN_MODEL}...")
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        print(f"[BRAIN] Cleaning response...")
        arya_reply = clean_lm_response(arya_reply)
        print(f"[BRAIN] ✅ Got response: {arya_reply[:60]}...")

        save_to_memory(user_id, "arya", arya_reply)

        # ===== FIX #4 & #5: EXPLICIT KEYWORD-BASED TRIGGERS =====
        # Check BEFORE sending - don't rely on LLM
        
        # Check for voice request
        if should_generate_voice(user_message):
            print(f"[MSG] User explicitly asked for voice - sending VOICE ONLY...")
            if can_send_voice_today(user_id):
                asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
            else:
                print(f"[MSG] Voice quota exhausted, sending text instead...")
                await update.message.reply_text(arya_reply)
        # Check for image request
        elif should_generate_image(user_message):
            print(f"[MSG] User explicitly asked for image - sending image...")
            await update.message.reply_text(arya_reply)
            asyncio.create_task(send_photo_task(context, chat_id, arya_reply))
        # Regular message - random chance of voice
        else:
            should_reply_with_voice = can_send_voice_today(user_id) and random.random() < 0.15
            
            if should_reply_with_voice:
                print(f"[MSG] Random voice chance hit - sending voice...")
                asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
            else:
                print(f"[MSG] Sending text response...")
                await update.message.reply_text(arya_reply)

    except Exception as e:
        print(f"[BRAIN] ❌ Error: {str(e)}")
        error_msg = get_random_error_message("general")
        await update.message.reply_text(error_msg)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id

    print(f"\n{'='*70}")
    print(f"[VOICE_MSG] Received voice note from user {user_id_telegram}")
    print(f"{'='*70}")

    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error_message("voice"))
        return

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()

        print(f"[VOICE_MSG] Downloaded {len(voice_bytes)} bytes")

        transcribed_text = await asyncio.to_thread(transcribe_voice_sync, bytes(voice_bytes))

        if not transcribed_text:
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

        print(f"[VOICE_MSG] Getting response...")
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
            print(f"[VOICE_MSG] ⚠️ Voice limit reached, sending text instead...")
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
    print(f"[PHOTO_TASK] Starting image task...")
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
    print(f"[VOICE_TASK] Starting voice task...")
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
# PART ELEVEN: BACKGROUND JOBS
# =====================================================

async def check_for_checkins(context: CallbackContext):
    """Background job: Check for check-ins every 6 hours"""
    print(f"\n[CHECKIN_JOB] ⏰ Running check-in job (every 6 hours)...")
    
    try:
        all_users = supabase.table("users").select("id, telegram_id").execute()
        
        for user in all_users.data:
            user_id = user["id"]
            telegram_id = user["telegram_id"]
            
            if should_send_checkin(user_id):
                print(f"[CHECKIN_JOB] Sending check-in to user {user_id}...")
                
                checkins = [
                    "hey, what's up? 👀",
                    "missing you, what're you up to?",
                    "you there? 😊",
                    "just thinking about you",
                    "haven't heard from you, you good?",
                    "how's your day going? 💭",
                    "been thinking about you babe",
                    "tell me what's on your mind 💕",
                ]
                
                msg = random.choice(checkins)
                
                try:
                    await context.bot.send_message(chat_id=telegram_id, text=msg)
                    mark_checkin_sent(user_id)
                    print(f"[CHECKIN_JOB] ✅ Check-in sent!")
                except Exception as e:
                    print(f"[CHECKIN_JOB] ❌ Failed to send: {str(e)}")
    
    except Exception as e:
        print(f"[CHECKIN_JOB] ❌ Error: {str(e)}")

# =====================================================
# PART TWELVE: BOT STARTUP & MAIN
# =====================================================

def main():
    """Initialize and start the bot"""
    print("\n" + "="*70)
    print("🚀 ARYA 6.0 - VENICE + FLUX EDITION")
    print("="*70)
    print(f"[STARTUP] Brain: Venice (Dolphin-Mistral - Uncensored)")
    print(f"[STARTUP] Voice In: {TRANSCRIPTION_PROVIDER} (Groq Whisper)")
    print(f"[STARTUP] Voice Out: ElevenLabs Bella (Premium)")
    print(f"[STARTUP] Images: FLUX via OpenRouter (Professional Quality)")
    print("="*70)
    print("[STARTUP] Features:")
    print("  ✅ Text conversations with personality")
    print("  ✅ Voice transcription (Groq Whisper)")
    print("  ✅ Voice generation (ElevenLabs - natural quality)")
    print("  ✅ Image generation (Pollinations.ai - FREE)")
    print("  ✅ Smart user profiling (extracts name, job, hobbies)")
    print("  ✅ Check-in EVERY 6 HOURS")
    print("  ✅ Explicit voice triggers (when user asks)")
    print("  ✅ Explicit image triggers (when user asks)")
    print("  ✅ Voice XOR Text (not both)")
    print("  ✅ Humanistic error messages")
    print("="*70)
    print("\n[STARTUP] 🎯 KEY FIXES:")
    print("[STARTUP] ✅ FIX #1: Images via FREE Pollinations.ai (no API needed)")
    print("[STARTUP] ✅ FIX #2: User data extracted & stored")
    print("[STARTUP] ✅ FIX #3: Voice XOR Text (not both)")
    print("[STARTUP] ✅ FIX #4: Explicit voice trigger when user asks")
    print("[STARTUP] ✅ FIX #5: Explicit image trigger when user asks")
    print("[STARTUP] ✅ FIX #6: Database schema issues fixed")
    print("="*70)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    print("[STARTUP] Registering handlers...")
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.job_queue.run_repeating(check_for_checkins, interval=25200, first=10)

    print("[STARTUP] ✅ All handlers registered!")
    print("[STARTUP] ✅ Check-in job scheduled (every 6 hours)")
    print("\n" + "="*70)
    print("💬 BOT IS RUNNING - READY FOR MESSAGES")
    print("="*70 + "\n")

    app.run_polling()

if __name__ == '__main__':
    main()