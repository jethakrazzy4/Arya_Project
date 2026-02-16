"""
=====================================================
ARYA 4.0 - FINAL COMPLETE VERSION (REPLICATE IMAGE)
=====================================================
All fixes included:
✅ Unbuffered output for Railway logs
✅ Response cleaning (no LLM thinking)
✅ No duplicate voices
✅ Daily check-ins
✅ Onboarding questions
✅ Humanistic errors
✅ Replicate FLUX image generation
=====================================================
"""

import asyncio
import os
import requests
import json
import urllib.parse
import random
import replicate
from io import BytesIO
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List

# External packages
from gtts import gTTS
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackContext
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

load_dotenv()

import os
SOULS = {
    'female': 'soul_female.txt',
    'male': 'soul_male.txt',
    'non-binary': 'soul_nb.txt'
}

def get_personality(gender='female'):
    """Load the right personality file based on gender"""
    try:
        filename = SOULS.get(gender, 'soul_female.txt')
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "You are Arya, a 24-year-old woman."

import sys
sys.stdout.reconfigure(line_buffering=True)

# =====================================================
# PART ONE: CENTRALIZED API CONFIGURATION
# =====================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

BRAIN_PROVIDER = "openrouter"
BRAIN_API_KEY = os.getenv("OPENROUTER_KEY")
BRAIN_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "deepseek/deepseek-v3.2"

TRANSCRIPTION_PROVIDER = "groq"
TRANSCRIPTION_API_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPTION_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3"

TTS_PROVIDER = "gtts"

# 🔥 IMAGE NOW USES REPLICATE
IMAGE_PROVIDER = "replicate"
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("\n" + "="*70)
print("[STARTUP] ARYA 4.0 - REPLICATE IMAGE VERSION")
print("="*70)
print("[STARTUP] Verifying API keys...")
assert TELEGRAM_TOKEN
assert BRAIN_API_KEY
assert TRANSCRIPTION_API_KEY
assert SUPABASE_URL
assert SUPABASE_KEY
assert REPLICATE_API_KEY
print("[STARTUP] ✅ All API keys verified!")

brain_client = OpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
transcription_client = OpenAI(base_url=TRANSCRIPTION_BASE_URL, api_key=TRANSCRIPTION_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[STARTUP] ✅ All clients initialized!")

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
# PART THREE: ONBOARDING QUESTIONS
# =====================================================

ONBOARDING_QUESTIONS = [
    {
        "id": 1,
        "question": "what's your name btw? i don't think i asked 😅",
        "field": "user_name",
        "day": 1
    },
    {
        "id": 2,
        "question": "so what do you do for work? or are you studying?",
        "field": "user_job",
        "day": 2
    },
    {
        "id": 3,
        "question": "what kinds of things do you like to do? hobbies and stuff?",
        "field": "user_hobbies",
        "day": 3
    },
]

# =====================================================
# IMAGE GENERATION (REPLICATE FLUX)
# =====================================================

def generate_image_sync(prompt: str) -> Optional[BytesIO]:
    print(f"[IMAGE] Starting Replicate generation...")

    try:
        clean_prompt = prompt.strip('[]').replace('[', '').replace(']', '')

        MANDATORY_LOOK = (
            "24-year-old woman, sharp chin-length dark hair bob, "
            "expressive dark eyes, natural realistic skin texture, athletic build"
        )

        enhanced_prompt = (
            f"RAW professional DSLR photo of {MANDATORY_LOOK}, "
            f"{clean_prompt}, realistic skin texture, natural lighting, "
            f"85mm lens, shallow depth of field, ultra-detailed"
        )

        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": enhanced_prompt,
                "num_outputs": 1,
                "height": 1024,
                "width": 1024,
            }
        )

        if output and len(output) > 0:
            image_url = output[0]
            img_response = requests.get(image_url, timeout=45)

            if img_response.status_code == 200:
                img_data = BytesIO(img_response.content)
                img_data.seek(0)
                img_data.name = "arya_capture.jpg"
                print("[IMAGE] ✅ Image generated!")
                return img_data

        print("[IMAGE] ❌ No image returned")
        return None

    except Exception as e:
        print(f"[IMAGE] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART FIVE: VOICE TRANSCRIPTION ENGINE
# =====================================================

def transcribe_voice_sync(voice_file_bytes: bytes) -> Optional[str]:
    """Transcribe voice using configured provider"""
    print(f"[VOICE_IN] Transcription starting ({len(voice_file_bytes)} bytes)...")
    
    try:
        audio_file = BytesIO(voice_file_bytes)
        audio_file.name = "voice.ogg"

        print(f"[VOICE_IN] Calling {TRANSCRIPTION_PROVIDER} API...")
        transcription = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="text"
        )

        transcribed_text = transcription.strip()
        print(f"[VOICE_IN] ✅ Transcribed: '{transcribed_text[:60]}...'")
        return transcribed_text

    except Exception as e:
        print(f"[VOICE_IN] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART SIX: VOICE GENERATION ENGINE (TTS)
# =====================================================

def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """Generate voice using configured provider"""
    print(f"[VOICE_OUT] Generating voice ({len(text)} chars)...")
    
    try:
        clean_text = text.replace("*", "").replace('"', "").replace("_", "")
        
        print(f"[VOICE_OUT] Using {TTS_PROVIDER} for TTS...")
        tts = gTTS(text=clean_text, lang='en', tld='co.uk')

        voice_io = BytesIO()
        tts.write_to_fp(voice_io)
        voice_io.seek(0)
        voice_io.name = 'arya_voice.mp3'

        print(f"[VOICE_OUT] ✅ Voice generated successfully!")
        return voice_io

    except Exception as e:
        print(f"[VOICE_OUT] ❌ Error: {str(e)}")
        return None

# =====================================================
# PART SEVEN: DATABASE & MEMORY FUNCTIONS
# =====================================================

def get_or_create_user(telegram_id: int) -> Optional[str]:
    """Get existing user or create new one"""
    print(f"[DB] Looking up user {telegram_id}...")
    
    try:
        response = supabase.table("users").select("id").filter(
            "telegram_id", "eq", telegram_id
        ).execute()

        if response.data and len(response.data) > 0:
            user_id = response.data[0]["id"]
            print(f"[DB] ✅ User found: {user_id}")
            return user_id

        print(f"[DB] Creating new user...")
        new_user = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "name": f"User{telegram_id}",
            "subscription_tier": "free",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()

        if not new_user.data:
            print(f"[DB] ❌ Failed to create user")
            return None

        user_id = new_user.data[0]["id"]
        
        print(f"[DB] Creating user profile...")
        supabase.table("user_profiles").insert({
            "user_id": user_id,
            "personality_choice": "female",
            "created_at": datetime.now().isoformat()
        }).execute()

        print(f"[DB] ✅ New user created: {user_id}")
        return user_id

    except Exception as e:
        print(f"[DB] ❌ Error: {str(e)}")
        return None

def save_to_memory(user_id: str, sender: str, text: str) -> bool:
    """Save message to conversation history"""
    try:
        if not user_id:
            print(f"[MEMORY] ⚠️ user_id is None!")
            return False
        
        response = supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": sender,
            "message": text[:2000],
            "created_at": datetime.now().isoformat()
        }).execute()

        if response.data and len(response.data) > 0:
            print(f"[MEMORY] ✅ Message saved from {sender}")
            return True
        else:
            print(f"[MEMORY] ⚠️ Insert returned no data")
            return False

    except Exception as e:
        print(f"[MEMORY] ❌ Error: {str(e)}")
        return False

def update_user_profile(user_id: str, updates: Dict) -> bool:
    """Update user profile"""
    try:
        if not user_id:
            print(f"[PROFILE] ⚠️ user_id is None!")
            return False
        
        print(f"[PROFILE] Updating: {list(updates.keys())}")
        
        response = supabase.table("user_profiles").update(updates).filter(
            "user_id", "eq", user_id
        ).execute()

        if response.data and len(response.data) > 0:
            print(f"[PROFILE] ✅ Updated")
            return True
        else:
            print(f"[PROFILE] ⚠️ Update might have failed")
            return False

    except Exception as e:
        print(f"[PROFILE] ❌ Error: {str(e)}")
        return False

def get_user_profile(user_id: str) -> Optional[Dict]:
    """Get user's profile info"""
    try:
        if not user_id:
            print(f"[PROFILE] ⚠️ user_id is None!")
            return None

        response = supabase.table("user_profiles").select("*").filter(
            "user_id", "eq", user_id
        ).execute()

        if response.data and len(response.data) > 0:
            print(f"[PROFILE] ✅ Profile loaded")
            return response.data[0]
        
        print(f"[PROFILE] Creating new profile...")
        supabase.table("user_profiles").insert({
            "user_id": user_id,
            "personality_choice": "female",
            "created_at": datetime.now().isoformat()
        }).execute()
        return None

    except Exception as e:
        print(f"[PROFILE] ❌ Error: {str(e)}")
        return None

def search_old_memories(user_id: str, query: str, limit: int = 3) -> str:
    """Search past conversations"""
    try:
        print(f"[MEMORY] Searching for: {query}")
        response = supabase.table("conversations").select("message").filter(
            "user_id", "eq", user_id
        ).filter(
            "message", "ilike", f"%{query}%"
        ).order("created_at", desc=True).limit(limit).execute()

        if response.data:
            result = " | ".join([r["message"] for r in response.data])
            print(f"[MEMORY] ✅ Found {len(response.data)} results")
            return result
        
        print(f"[MEMORY] ⚠️ No results found")
        return ""

    except Exception as e:
        print(f"[MEMORY] ❌ Error: {str(e)}")
        return ""

def get_conversation_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent conversation history"""
    try:
        response = supabase.table("conversations").select("sender, message").filter(
            "user_id", "eq", user_id
        ).order("created_at", desc=True).limit(limit).execute()

        history = list(reversed(response.data))
        print(f"[MEMORY] ✅ Loaded {len(history)} messages")
        return history

    except Exception as e:
        print(f"[MEMORY] ❌ Error: {str(e)}")
        return []

# =====================================================
# PART EIGHT: VOICE & CHECK-IN TRACKING
# =====================================================

def can_send_voice_today(user_id: str) -> bool:
    """Check if we should send voice today (one per user per day)"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True

        if profile.get("voice_sent_today") and profile.get("last_voice_sent_at"):
            sent_date = datetime.fromisoformat(profile["last_voice_sent_at"]).date()
            if sent_date == date.today():
                print(f"[VOICE_LIMIT] ⚠️ Voice already sent today for user {user_id}")
                return False

        print(f"[VOICE_LIMIT] ✅ Can send voice today")
        return True

    except Exception as e:
        print(f"[VOICE_LIMIT] ❌ Error: {str(e)}")
        return False

def mark_voice_sent(user_id: str) -> bool:
    """Mark voice sent today"""
    try:
        print(f"[VOICE_LIMIT] Marking voice sent for user {user_id}")
        update_user_profile(user_id, {
            "voice_sent_today": True,
            "last_voice_sent_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        return True
    except Exception as e:
        print(f"[VOICE_LIMIT] ❌ Error: {str(e)}")
        return False

def should_send_checkin(user_id: str) -> bool:
    """Check if should send 24-hour check-in"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return False

        last_msg = profile.get("last_message_from_user")
        if not last_msg:
            return False

        last_msg_time = datetime.fromisoformat(last_msg)
        now = datetime.now()

        if (now - last_msg_time).total_seconds() < 86400:
            print(f"[CHECKIN] ⚠️ User messaged recently")
            return False

        last_checkin = profile.get("last_checkin_sent")
        if last_checkin:
            checkin_date = datetime.fromisoformat(last_checkin).date()
            if checkin_date == date.today():
                print(f"[CHECKIN] ⚠️ Check-in already sent today")
                return False

        print(f"[CHECKIN] ✅ Should send check-in!")
        return True

    except Exception as e:
        print(f"[CHECKIN] ❌ Error: {str(e)}")
        return False

def mark_checkin_sent(user_id: str) -> bool:
    """Mark check-in sent"""
    try:
        print(f"[CHECKIN] Marking check-in sent for user {user_id}")
        update_user_profile(user_id, {
            "last_checkin_sent": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        return True
    except Exception as e:
        print(f"[CHECKIN] ❌ Error: {str(e)}")
        return False

# =====================================================
# PART NINE: ONBOARDING QUESTION LOGIC
# =====================================================

def get_next_onboarding_question(user_id: str) -> Optional[str]:
    """Get next onboarding question"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return None

        last_q_date = profile.get("last_question_date")
        if last_q_date:
            last_q_date = datetime.fromisoformat(last_q_date).date() if isinstance(last_q_date, str) else last_q_date
            if last_q_date < date.today():
                print(f"[ONBOARD] Resetting daily question counter")
                update_user_profile(user_id, {"questions_asked_today": 0})
                profile["questions_asked_today"] = 0

        if profile.get("questions_asked_today", 0) >= 3:
            print(f"[ONBOARD] ⚠️ Hit daily question limit")
            return None

        for q in ONBOARDING_QUESTIONS:
            field = q["field"]
            if not profile.get(field):
                print(f"[ONBOARD] Next question: {field}")
                return q["question"]

        print(f"[ONBOARD] ✅ Onboarding complete!")
        update_user_profile(user_id, {"onboarding_complete": True})
        return None

    except Exception as e:
        print(f"[ONBOARD] ❌ Error: {str(e)}")
        return None

def increment_daily_questions(user_id: str) -> bool:
    """Increment daily question counter"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return False

        new_count = profile.get("questions_asked_today", 0) + 1
        update_user_profile(user_id, {
            "questions_asked_today": new_count,
            "last_question_date": date.today().isoformat()
        })
        return True

    except Exception as e:
        print(f"[ONBOARD] ❌ Error: {str(e)}")
        return False

# =====================================================
# PART TEN: MESSAGE HANDLERS
# =====================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user_id_telegram = update.effective_user.id
    user_text = update.message.text
    chat_id = update.effective_chat.id

    print(f"\n{'='*70}")
    print(f"[USER_MSG] User {user_id_telegram}: {user_text[:50]}...")
    print(f"{'='*70}")

    user_id = get_or_create_user(user_id_telegram)
    
    # ✅ NEW: Ask for personality on first message
    profile = get_user_profile(user_id)
    if not profile.get("personality_choice"):
        if not profile.get("first_message_seen"):
            # First time - ask personality question
            update_user_profile(user_id, {"first_message_seen": True})
            await update.message.reply_text("hey 👀 am i your girl, guy, or something else? (just say: girl/guy/other)")
            return
        else:
            # They answered the question
            text_lower = user_text.lower().strip()
            if text_lower in ['girl', 'female', 'woman', 'f']:
                update_user_profile(user_id, {"personality_choice": "female"})
                await update.message.reply_text("perfect 😊")
                return
            elif text_lower in ['guy', 'male', 'man', 'm']:
                update_user_profile(user_id, {"personality_choice": "male"})
                await update.message.reply_text("nice 😎")
                return
            elif text_lower in ['other', 'non-binary', 'nb']:
                update_user_profile(user_id, {"personality_choice": "non-binary"})
                await update.message.reply_text("cool 🌈")
                return
            else:
                await update.message.reply_text("hey, just say: girl/guy/other")
                return

    save_to_memory(user_id, "user", user_text)
    update_user_profile(user_id, {
        "last_message_from_user": datetime.now().isoformat()
    })

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    past_context = ""
    if any(k in user_text.lower() for k in ["remember", "told you", "about", "said"]):
        past_context = f"\nRELEVANT PAST INFO: {search_old_memories(user_id, user_text[:20])}"

    print(f"[BRAIN] Loading personality...")
    profile = get_user_profile(user_id)
    personality = profile.get("personality_choice", "female") if profile else "female"
    soul_content = get_personality(personality)
    print(f"[BRAIN] Using {personality} personality")

    history = get_conversation_history(user_id, limit=10)

    profile = get_user_profile(user_id)
    user_context = ""
    if profile and profile.get("user_name"):
        user_context += f"\nUSER'S NAME: {profile['user_name']}"
    if profile and profile.get("user_job"):
        user_context += f"\nUSER'S JOB: {profile['user_job']}"
    if profile and profile.get("user_hobbies"):
        user_context += f"\nUSER'S HOBBIES: {profile['user_hobbies']}"

    messages = [{"role": "system", "content": soul_content + user_context + past_context}]
    for record in history:
        role = "user" if record["sender"] == "user" else "assistant"
        messages.append({"role": role, "content": record["message"]})

    print(f"[BRAIN] Calling {BRAIN_MODEL}...")
    try:
        response = brain_client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        print(f"[BRAIN] ✅ Raw response received, cleaning...")
        arya_reply = clean_lm_response(arya_reply)
        print(f"[BRAIN] ✅ Cleaned response: {arya_reply[:60]}...")

        save_to_memory(user_id, "arya", arya_reply)

        sent_any_text = False
        parts = arya_reply.split('\n\n')
        
        for part in parts[:3]:
            if not part.strip():
                continue

            if "IMAGE_PROMPT:" in part:
                prompt = part.split("IMAGE_PROMPT:")[1].strip()
                print(f"[MSG] Detected image request, queuing...")
                asyncio.create_task(send_photo_task(context, chat_id, prompt))
                continue
            
            print(f"[MSG] Sending text message...")
            await update.message.reply_text(part.strip())
            sent_any_text = True

        next_question = get_next_onboarding_question(user_id)
        if next_question:
            print(f"[MSG] Adding onboarding question...")
            await update.message.reply_text(next_question)
            increment_daily_questions(user_id)
            sent_any_text = True

        print(f"[MSG] Text sent: {sent_any_text}, checking voice eligibility...")
        if not sent_any_text and can_send_voice_today(user_id):
            if random.random() < 0.2:
                print(f"[MSG] Queueing voice note...")
                asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))

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

        save_to_memory(user_id, "user", f"[voice] {transcribed_text}")
        update_user_profile(user_id, {
            "last_message_from_user": datetime.now().isoformat()
        })

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        profile = get_user_profile(user_id)
        personality = profile.get("personality_choice", "female") if profile else "female"
        soul_content = get_personality(personality)
        print(f"[VOICE_MSG] Using {personality} personality")

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

        print(f"[MSG] Sending text reply to voice...")
        await update.message.reply_text(arya_reply)

        if can_send_voice_today(user_id):
            print(f"[VOICE_MSG] Sending voice reply (user initiated voice)...")
            asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
        else:
            print(f"[VOICE_MSG] ⚠️ Voice limit reached for today, only sent text")

    except Exception as e:
        print(f"[VOICE_MSG] ❌ Error: {str(e)}")
        error_msg = get_random_error_message("voice")
        await update.message.reply_text(error_msg)

# =====================================================
# PART ELEVEN: ASYNC TASK HANDLERS
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
# PART TWELVE: BACKGROUND JOBS
# =====================================================

async def check_for_checkins(context: CallbackContext):
    """Background job: Check for 24-hour check-ins"""
    print(f"\n[CHECKIN_JOB] ⏰ Running check-in job...")
    
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
# PART THIRTEEN: BOT STARTUP & MAIN
# =====================================================

def main():
    """Initialize and start the bot"""
    print("\n" + "="*70)
    print("🚀 ARYA 4.0 - FINAL COMPLETE VERSION")
    print("="*70)
    print(f"[STARTUP] Brain: {BRAIN_MODEL}")
    print(f"[STARTUP] Voice In: {TRANSCRIPTION_PROVIDER}")
    print(f"[STARTUP] Voice Out: {TTS_PROVIDER}")
    print(f"[STARTUP] Images: {IMAGE_PROVIDER}")
    print("="*70)
    print("[STARTUP] Features:")
    print("  ✅ Text conversations with personality")
    print("  ✅ Voice transcription (Groq Whisper)")
    print("  ✅ Voice generation (gTTS)")
    print("  ✅ Image generation (Pollinations.ai)")
    print("  ✅ User profiling & onboarding")
    print("  ✅ Daily 24-hour check-ins")
    print("  ✅ One voice note per user per day")
    print("  ✅ Humanistic error messages")
    print("  ✅ Detailed logging for Railway")
    print("  ✅ Response cleaning (no LLM thinking)")
    print("="*70)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    print("[STARTUP] Registering handlers...")
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.job_queue.run_repeating(check_for_checkins, interval=7200, first=10)

    print("[STARTUP] ✅ All handlers registered!")
    print("\n" + "="*70)
    print("💬 BOT IS RUNNING - READY FOR MESSAGES")
    print("="*70 + "\n")

    app.run_polling()

if __name__ == '__main__':
    main()