import asyncio, os, requests
from io import BytesIO
from gtts import gTTS
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
import json
import urllib.parse

# Load environment variables
load_dotenv()

# ==========================================
# 1. THE KEYS & CONFIG
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
#HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_ID = "tngtech/deepseek-r1t2-chimera:free"

# ❌ REMOVED: PROXIES line - Railway doesn't need proxy!
# ✅ Railway can access external APIs directly

# Verify all keys are loaded
assert TELEGRAM_TOKEN, "ERROR: TELEGRAM_TOKEN not found"
assert OPENROUTER_KEY, "ERROR: OPENROUTER_KEY not found"
#assert HF_TOKEN, "ERROR: HF_TOKEN not found"
assert SUPABASE_URL, "ERROR: SUPABASE_URL not found"
assert SUPABASE_KEY, "ERROR: SUPABASE_KEY not found"

# Initialize clients
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. INTERNAL ENGINES (Voice & Eyes) - FIXED
# ==========================================

def generate_voice_sync(text):
    """Generate voice and return as BytesIO (in-memory, no disk save)"""
    try:
        clean_text = text.replace("*", "").replace('"', "")
        tts = gTTS(text=clean_text, lang='en', tld='co.uk')
        
        # ✅ FIXED: Use BytesIO instead of saving to disk
        voice_io = BytesIO()
        tts.write_to_fp(voice_io)
        voice_io.seek(0)
        voice_io.name = 'arya_voice.mp3'  # Set filename for Telegram
        
        return voice_io
    except Exception as e:
        print(f"Voice Error: {e}")
        return None

def generate_image_sync(prompt):
    """Generate image using Pollinations.AI (FREE, uncensored)"""
    import urllib.parse
    
    MANDATORY_LOOK = "24-year-old woman, sharp chin-length dark hair bob, expressive dark eyes, natural realistic skin texture, athletic build"
    full_prompt = f"Cinematic photo of {MANDATORY_LOOK}, {prompt}, high quality, 8k, sharp focus, vibrant colors"
    
    # URL encode the prompt
    encoded_prompt = urllib.parse.quote(full_prompt)
    
    # Pollinations.AI - FREE image generation
    API_URL = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux-realism&width=1536&height=1536&model=flux&nologo=true&enhance=true"
    
    try:
        print(f"🎨 Generating image with Pollinations.AI...")
        response = requests.get(API_URL, timeout=30)
        
        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img_data.seek(0)
            img_data.name = 'arya_capture.jpg'
            print(f"✅ Image generated successfully (FREE)")
            return img_data
        else:
            print(f"❌ Image API returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Image API Error: {e}")
        return None

# ==========================================
# 3. DATABASE & MEMORY (Supabase)
# ==========================================

def get_or_create_user(telegram_id):
    """Get existing user or create new one"""
    try:
        # Check if user exists
        response = supabase.table("users").select("id").filter(
            "telegram_id", "eq", telegram_id
        ).execute()

        if response.data:
            return response.data[0]["id"]
        
        # Create new user
        new_user = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "name": f"User{telegram_id}",
            "subscription_tier": "free",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        
        return new_user.data[0]["id"]
    except Exception as e:
        print(f"User creation error: {e}")
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
        print(f"Memory save error: {e}")

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
        print(f"Memory search error: {e}")
        return ""

def get_conversation_history(user_id, limit=10):
    """Get recent conversation history"""
    try:
        response = supabase.table("conversations").select("sender, message").filter(
            "user_id", "eq", user_id
        ).order("created_at", desc=True).limit(limit).execute()
        
        return list(reversed(response.data))
    except Exception as e:
        print(f"History fetch error: {e}")
        return []

# ==========================================
# 4. HANDLERS - IMPROVED
# ==========================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler"""
    user_id_telegram = update.effective_user.id
    user_text = update.message.text
    chat_id = update.effective_chat.id

    # Get or create user in database
    user_id = get_or_create_user(user_id_telegram)
    
    if not user_id:
        await update.message.reply_text("Error: Could not create user session. Please try again.")
        return

    # Save user message
    save_to_memory(user_id, "user", user_text)
    
    # Show typing indicator
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

    # Build message history from database
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
        
        # Save Arya's response
        save_to_memory(user_id, "arya", arya_reply)

        # Process response for images and voice
        parts = arya_reply.split('\n\n')
        
        for part in parts[:3]:  # Max 3 message bubbles
            if not part.strip():
                continue
                
            # Check for image request
            if "IMAGE_PROMPT:" in part:
                prompt = part.split("IMAGE_PROMPT:")[1].strip()
                asyncio.create_task(send_photo_task(context, chat_id, prompt))
                continue
            
            # Check for voice request (optional - if user asks)
            if "VOICE:" in part.lower() or "send voice" in user_text.lower():
                asyncio.create_task(send_voice_task(context, chat_id, part))
                continue
            
            # Send text message
            await update.message.reply_text(part.strip())
    
    except Exception as e:
        print(f"Brain Error: {e}")
        await update.message.reply_text("Sorry, I'm having trouble thinking right now. Please try again.")

async def send_photo_task(context, chat_id, prompt):
    """Generate and send image"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        
        # Generate image in thread pool (blocks)
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

async def send_voice_task(context, chat_id, text):
    """Generate and send voice message"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
        
        # Generate voice in thread pool
        voice = await asyncio.to_thread(generate_voice_sync, text)
        
        if voice:
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=voice
            )
            print(f"✅ Voice sent to chat {chat_id}")
        else:
            print(f"❌ Voice generation failed for chat {chat_id}")
    except Exception as e:
        print(f"❌ Voice send error: {e}")

# ==========================================
# 5. START
# ==========================================

if __name__ == '__main__':
    print("🚀 Arya 2.9 Cloud Edition - Starting...")
    print("✅ Proxy removed (Railway-compatible)")
    print("✅ Voice using BytesIO (no disk writes)")
    print("✅ Image generation fixed")
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("🤖 Bot is running. Press Ctrl+C to stop.")
    app.run_polling()