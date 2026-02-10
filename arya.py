import asyncio, os, requests
from io import BytesIO
from gtts import gTTS
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timedelta
import json
import urllib.parse
import random

# Load environment variables
load_dotenv()

# ==========================================
# 1. THE KEYS & CONFIG
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # New: For Whisper transcription
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")

# Verify all keys are loaded
assert TELEGRAM_TOKEN, "ERROR: TELEGRAM_TOKEN not found"
assert OPENROUTER_KEY, "ERROR: OPENROUTER_KEY not found"
assert GROQ_API_KEY, "ERROR: GROQ_API_KEY not found"
assert SUPABASE_URL, "ERROR: SUPABASE_URL not found"
assert SUPABASE_KEY, "ERROR: SUPABASE_KEY not found"
assert POLLINATIONS_API_KEY, "ERROR: POLLINATIONS_API_KEY not found"

# Initialize clients
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. INTERNAL ENGINES (Voice & Eyes)
# ==========================================

def generate_voice_sync(text):
    """
    Generate voice using gTTS (Google Text-to-Speech)
    Returns BytesIO object (in-memory, no disk save)
    """
    try:
        # Clean text (remove markdown and special characters)
        clean_text = text.replace("*", "").replace('"', "").replace("_", "")
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', tld='co.uk')
        
        # Save to memory (not disk)
        voice_io = BytesIO()
        tts.write_to_fp(voice_io)
        voice_io.seek(0)
        voice_io.name = 'arya_voice.mp3'
        
        print(f"✅ Voice generated ({len(clean_text)} chars)")
        return voice_io
        
    except Exception as e:
        print(f"❌ Voice Error: {e}")
        return None

def transcribe_voice_sync(voice_file_bytes):
    """
    Transcribe voice using Groq Whisper (FREE, fast, accurate)
    
    Args:
        voice_file_bytes: Audio file as bytes
    
    Returns:
        Transcribed text string
    """
    try:
        # Groq Whisper requires file-like object
        audio_file = BytesIO(voice_file_bytes)
        audio_file.name = "voice.ogg"  # Telegram sends .ogg files
        
        # Call Groq Whisper API
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",  # Best Whisper model
            file=audio_file,
            response_format="text"
        )
        
        transcribed_text = transcription.strip()
        print(f"✅ Voice transcribed: '{transcribed_text[:50]}...'")
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Transcription Error: {e}")
        return None

def generate_image_sync(prompt):
    """
    Generate image using Pollinations.AI (FREE, uncensored)
    """
    MANDATORY_LOOK = (
        "24-year-old woman, sharp chin-length dark hair bob, "
        "expressive dark eyes, natural realistic skin texture, athletic build"
    )

    full_prompt = (
        f"RAW professional DSLR photo of {MANDATORY_LOOK}, "
        f"{prompt}, realistic skin texture, natural lighting, "
        f"85mm lens, shallow depth of field, ultra-detailed"
    )

    encoded_prompt = urllib.parse.quote(full_prompt)

    API_URL = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=1024&model=flux&nologo=true&enhance=true"
    )

    try:
        print(f"🎨 Generating image...")
        response = requests.get(API_URL, timeout=45)

        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img_data.seek(0)
            img_data.name = "arya_capture.jpg"
            print(f"✅ Image generated")
            return img_data
        else:
            print(f"❌ Pollinations API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return None

# ==========================================
# 3. DATABASE & MEMORY (Supabase)
# ==========================================

def get_or_create_user(telegram_id):
    """Get existing user or create new one"""
    try:
        response = supabase.table("users").select("id").filter(
            "telegram_id", "eq", telegram_id
        ).execute()

        if response.data:
            return response.data[0]["id"]
        
        new_user = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "name": f"User{telegram_id}",
            "subscription_tier": "free",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        
        return new_user.data[0]["id"]
    except Exception as e:
        print(f"❌ User creation error: {e}")
        return None

def save_to_memory(user_id, sender, text):
    """Save message to Supabase"""
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": sender,
            "message": text,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        print(f"❌ Memory save error: {e}")

def search_old_memories(user_id, query, limit=3):
    """Search past conversations"""
    try:
        response = supabase.table("conversations").select("message").filter(
            "user_id", "eq", user_id
        ).filter(
            "message", "ilike", f"%{query}%"
        ).order("created_at", desc=True).limit(limit).execute()
        
        return " | ".join([r["message"] for r in response.data])
    except Exception as e:
        print(f"❌ Memory search error: {e}")
        return ""

def get_conversation_history(user_id, limit=10):
    """Get recent conversation history"""
    try:
        response = supabase.table("conversations").select("sender, message").filter(
            "user_id", "eq", user_id
        ).order("created_at", desc=True).limit(limit).execute()
        
        return list(reversed(response.data))
    except Exception as e:
        print(f"❌ History fetch error: {e}")
        return []

def should_send_voice_today(user_id):
    """
    Determine if Arya should send voice note today (randomly 1-2 times/day)
    
    Logic:
    - Check if voice was sent in last 12 hours
    - If not, 30% chance to send voice
    - Ensures 1-2 voice notes per day naturally
    """
    try:
        # Get last voice message from Arya
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).isoformat()
        
        response = supabase.table("conversations").select("created_at").filter(
            "user_id", "eq", user_id
        ).filter(
            "sender", "eq", "arya_voice"  # Track voice separately
        ).filter(
            "created_at", "gte", twelve_hours_ago
        ).execute()
        
        # If voice sent in last 12 hours, don't send again
        if response.data:
            return False
        
        # 30% chance to send voice (results in ~1-2 per day)
        return random.random() < 0.3
        
    except Exception as e:
        print(f"❌ Voice check error: {e}")
        # Default: 10% chance if error
        return random.random() < 0.1

def mark_voice_sent(user_id):
    """Mark that voice was sent (for tracking)"""
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": "arya_voice",  # Special marker
            "message": "[voice_sent]",
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        print(f"❌ Voice marker error: {e}")

# ==========================================
# 4. HANDLERS
# ==========================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler (text messages)"""
    user_id_telegram = update.effective_user.id
    user_text = update.message.text
    chat_id = update.effective_chat.id

    user_id = get_or_create_user(user_id_telegram)
    
    if not user_id:
        await update.message.reply_text("Error: Could not create user session. Please try again.")
        return

    save_to_memory(user_id, "user", user_text)
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Build conversation context
    past_context = ""
    if any(k in user_text.lower() for k in ["remember", "apartment", "burnaby", "data pilot"]):
        past_context = f"\nRELEVANT PAST INFO: {search_old_memories(user_id, user_text[:10])}"

    # Load personality
    try:
        with open("soul.txt", "r", encoding="utf-8") as f:
            soul_content = f.read()
    except:
        soul_content = "You are Arya, a 24-year-old AI companion."

    # Build message history
    history = get_conversation_history(user_id, limit=10)
    
    messages = [{"role": "system", "content": soul_content + past_context}]
    for record in history:
        role = "user" if record["sender"] == "user" else "assistant"
        messages.append({"role": role, "content": record["message"]})

    # Get response from Arya
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        save_to_memory(user_id, "arya", arya_reply)

        # Process response
        parts = arya_reply.split('\n\n')
        
        for part in parts[:3]:
            if not part.strip():
                continue
                
            # Check for image request
            if "IMAGE_PROMPT:" in part:
                prompt = part.split("IMAGE_PROMPT:")[1].strip()
                asyncio.create_task(send_photo_task(context, chat_id, prompt))
                continue
            
            # Send text message
            await update.message.reply_text(part.strip())
        
        # RANDOM VOICE: Decide if Arya should send voice (1-2 times/day)
        if should_send_voice_today(user_id):
            print(f"🎤 Sending random voice note to user {user_id}")
            asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
    
    except Exception as e:
        print(f"❌ Brain Error: {e}")
        await update.message.reply_text("Sorry, I'm having trouble thinking right now. Please try again.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle voice messages from user
    1. Download voice file
    2. Transcribe using Groq Whisper
    3. Process as text message
    """
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    
    user_id = get_or_create_user(user_id_telegram)
    
    if not user_id:
        await update.message.reply_text("Error: Could not create user session.")
        return
    
    try:
        # Show that Arya is listening
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Download voice file from Telegram
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        print(f"🎙️ Received voice note from user {user_id} ({len(voice_bytes)} bytes)")
        
        # Transcribe using Groq Whisper (in thread pool - blocking operation)
        transcribed_text = await asyncio.to_thread(transcribe_voice_sync, bytes(voice_bytes))
        
        if not transcribed_text:
            await update.message.reply_text("Sorry, I couldn't hear that clearly. Can you try again?")
            return
        
        print(f"📝 Transcribed: '{transcribed_text}'")
        
        # Save transcription to memory
        save_to_memory(user_id, "user", f"[voice] {transcribed_text}")
        
        # Process as normal text message
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Load personality
        try:
            with open("soul.txt", "r", encoding="utf-8") as f:
                soul_content = f.read()
        except:
            soul_content = "You are Arya, a 24-year-old AI companion."
        
        # Build message history
        history = get_conversation_history(user_id, limit=10)
        
        messages = [{"role": "system", "content": soul_content}]
        for record in history:
            role = "user" if record["sender"] == "user" else "assistant"
            # Clean [voice] prefix for context
            msg = record["message"].replace("[voice] ", "")
            messages.append({"role": role, "content": msg})
        
        # Get Arya's response
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.9
        )
        arya_reply = response.choices[0].message.content or "..."
        
        save_to_memory(user_id, "arya", arya_reply)
        
        # Send text response
        await update.message.reply_text(arya_reply)
        
        # ALWAYS send voice back when user sends voice (feel natural)
        print(f"🎤 Sending voice reply to user voice note")
        asyncio.create_task(send_voice_task(context, chat_id, arya_reply, user_id))
        
    except Exception as e:
        print(f"❌ Voice handling error: {e}")
        await update.message.reply_text("Sorry, I had trouble with that voice note. Try again?")

async def send_photo_task(context, chat_id, prompt):
    """Generate and send image"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        
        photo = await asyncio.to_thread(generate_image_sync, prompt)
        
        if photo:
            await context.bot.send_photo(
                chat_id=chat_id, 
                photo=photo, 
                caption="for you... 😉"
            )
            print(f"✅ Photo sent to chat {chat_id}")
        else:
            print(f"❌ Photo generation failed for chat {chat_id}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="Sorry, couldn't generate the image right now. Try again?"
            )
    except Exception as e:
        print(f"❌ Photo send error: {e}")

async def send_voice_task(context, chat_id, text, user_id=None):
    """
    Generate and send voice message
    
    Args:
        context: Telegram context
        chat_id: Chat ID to send to
        text: Text to convert to speech
        user_id: User ID (for tracking random voice)
    """
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
        
        # Limit voice length (gTTS can handle max ~500 chars comfortably)
        voice_text = text[:500] if len(text) > 500 else text
        
        # Generate voice in thread pool
        voice = await asyncio.to_thread(generate_voice_sync, voice_text)
        
        if voice:
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=voice
            )
            print(f"✅ Voice sent to chat {chat_id}")
            
            # Mark voice sent (for random tracking)
            if user_id:
                mark_voice_sent(user_id)
        else:
            print(f"❌ Voice generation failed for chat {chat_id}")
            
    except Exception as e:
        print(f"❌ Voice send error: {e}")

# ==========================================
# 5. START
# ==========================================

if __name__ == '__main__':
    print("🚀 Arya 3.0 Voice Edition - Starting...")
    print("✅ Voice INPUT: Groq Whisper (FREE transcription)")
    print("✅ Voice OUTPUT: gTTS (random 1-2 times/day)")
    print("✅ Image generation: Pollinations.AI (FREE)")
    print("🤖 Arya is online and ready! ✨")
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Handle text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Handle voice messages (NEW!)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    print("💬 Bot is running. Send text or voice messages!")
    app.run_polling()