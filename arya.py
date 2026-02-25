"""
ARYA 4.0 - FINAL PRODUCTION VERSION
Premium AI Companion Service

FEATURES:
- /start command with personality selection (Arya/Alex/Aris)
- Mandatory onboarding: Name, Age, Timezone
- Full conversation history
- Image and voice support
- RLS-secured database
- Production logging
- TIME-AWARE responses (knows user's timezone)
- DUAL-BRAIN system (DeepSeek for roleplay, Perplexity for intelligence)

SCHEMA:
- user_name: Collected during onboarding
- user_age: Collected during onboarding  
- user_timezone: Collected during onboarding (for time-aware messages)
- personality_choice: Selected at /start
- All data stored securely in Supabase
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
import pytz

from gtts import gTTS
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackContext, CommandHandler, CallbackQueryHandler
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client

# =====================================================
# SECTION 1: CONFIG & INITIALIZATION
# =====================================================

load_dotenv()

import sys
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== TELEGRAM =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# ===== PRIMARY LLM (ROLEPLAY) =====
BRAIN_PROVIDER = "openrouter"
BRAIN_API_KEY = os.getenv("OPENROUTER_KEY")
BRAIN_BASE_URL = "https://openrouter.ai/api/v1"
BRAIN_MODEL = "deepseek/deepseek-v3.2"

# ===== SECONDARY LLM (INTELLIGENCE/NEWS) - NEW =====
INTELLIGENCE_PROVIDER = "openrouter"
INTELLIGENCE_API_KEY = os.getenv("OPENROUTER_KEY")  # Same key as primary
INTELLIGENCE_BASE_URL = "https://openrouter.ai/api/v1"
INTELLIGENCE_MODEL = "perplexity/sonar-pro"  # Has web search + current knowledge
INTELLIGENCE_ENABLED = True

# ===== VOICE TRANSCRIPTION =====
TRANSCRIPTION_PROVIDER = "groq"
TRANSCRIPTION_API_KEY = os.getenv("GROQ_API_KEY")
TRANSCRIPTION_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3"

# ===== VOICE GENERATION =====
TTS_PROVIDER = "gtts"

# ===== IMAGE GENERATION =====
IMAGE_PROVIDER = "eternal_ai"
ETERNAL_AI_API_KEY = os.getenv("ETERNAL_AI_API_KEY")
ETERNAL_AI_SUBMIT_URL = "https://agentic.eternalai.org/prompt"
ETERNAL_AI_POLL_URL = "https://agent-api.eternalai.org/result"
ETERNAL_AI_AGENT = "uncensored-imagine"
ETERNAL_AI_MAX_POLLS = 60
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

IMAGE_MANDATORY_LOOK = (
    "24-year-old woman, dark hair bob, dark eyes, athletic build. "
    "Natural lighting, photography. Real person."
)

# ===== DATABASE =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ===== PERSONALITY FILES =====
PERSONALITY_FILES = {
    'female': 'soul_female.txt',
    'male': 'soul_male.txt',
    'non-binary': 'soul_nb.txt'
}

PERSONALITIES_INFO = {
    'female': {'name': 'Arya', 'gender': 'Female'},
    'male': {'name': 'Alex', 'gender': 'Male'},
    'non-binary': {'name': 'Aris', 'gender': 'Non-binary'}
}

# ===== TIMEZONE VALIDATION =====
VALID_TIMEZONES = pytz.all_timezones
COMMON_TIMEZONES = {
    'utc': 'UTC',
    'est': 'America/New_York',
    'cst': 'America/Chicago',
    'mst': 'America/Denver',
    'pst': 'America/Los_Angeles',
    'gmt': 'Europe/London',
    'ist': 'Asia/Kolkata',
    'jst': 'Asia/Tokyo',
    'aest': 'Australia/Sydney',
}

# ===== ERROR MESSAGES =====
ERROR_MESSAGES = {
    "general": [
        "ugh my head hurts rn 😅", "i'm having a weird moment, can we try again?",
        "sorry brain fog happening, ask me later", "lowkey not thinking straight right now",
        "i'm kinda out of it, can we try again?", "head's all fuzzy, give me a sec",
        "idk what's wrong with me rn honestly", "my brain needs a coffee break lol",
        "having a weird technical hiccup, sorry", "signal's being dumb, try again?",
    ],
    "voice": [
        "couldn't quite hear that, sorry", "my ears are broken rn lol",
        "couldn't catch that, can you repeat?", "you're breaking up i think?",
        "didn't get that one, try again?",
    ],
    "image": [
        "can't find my camera rn 😅", "picture's not loading, try again?",
        "my phone's being weird, hold on", "camera's acting up, sorry babe",
        "can't get a good shot rn",
    ]
}

# ===== RESPONSE CLEANING =====
THINKING_PATTERNS = [
    "alright,", "first,", "she should", "the tone should", "this is",
    "since", "arya needs", "arya's", "the user", "let me", "i need to",
    "the response", "she needs", "this means", "so she", "aiming for",
    "prompt should", "visual prompt", "arya's reply", "wrapping up",
]

# ===== ONBOARDING QUESTIONS =====
ONBOARDING_QUESTIONS = [
    {"id": 1, "question": "what's your name btw? (first and last) 😊", "field": "user_name"},
    {"id": 2, "question": "and how old are you?", "field": "user_age"},
    {"id": 3, "question": "what's your timezone? (e.g., America/New_York or Europe/London or Asia/Tokyo) 🌍", "field": "user_timezone"},
]

# ===== INTELLIGENCE QUESTION KEYWORDS =====
INTELLIGENCE_KEYWORDS = [
    "news", "current", "today", "this week", "this month", "market", "stock",
    "price", "crypto", "bitcoin", "ethereum", "event", "happened", "breaking",
    "latest", "recent", "update", "weather", "sports score", "game", "match",
    "world", "politics", "election", "government", "what happened",
]

# ===== VERIFY CONFIG =====
def verify_config():
    """Verify all required API keys present"""
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
    logger.info(f"✅ Dual-brain enabled: {BRAIN_MODEL} (roleplay) + {INTELLIGENCE_MODEL} (intelligence)")

verify_config()

brain_client = OpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
intelligence_client = OpenAI(base_url=INTELLIGENCE_BASE_URL, api_key=INTELLIGENCE_API_KEY)
transcription_client = OpenAI(base_url=TRANSCRIPTION_BASE_URL, api_key=TRANSCRIPTION_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logger.info("✅ All API clients initialized!")


# =====================================================
# SECTION 2: DATABASE FUNCTIONS
# =====================================================

# ===== POST-ONBOARDING QUESTIONS =====
POST_ONBOARDING_QUESTIONS = [
    {"field": "job", "question": "What do you do for work? 😊"},
    {"field": "hobby1", "question": "Got any hobbies? Tell me one."},
    {"field": "hobby2", "question": "Another hobby? I'm curious."},
    {"field": "hobby3", "question": "And one more hobby? I love learning about you."},
    {"field": "relationship_status", "question": "Are you in a relationship or single? (just curious)"},
    {"field": "location", "question": "Where are you from?"},
    {"field": "interests", "question": "What are you passionate about?"},
]

def get_or_create_user(telegram_id: int) -> Optional[str]:
    """Get existing user or create new one"""
    try:
        result = supabase.table("users").select("id").eq("telegram_id", telegram_id).execute()
        if result.data:
            return result.data[0]["id"]
        
        logger.info(f"Creating new user: {telegram_id}")
        user_result = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if user_result.data:
            return user_result.data[0]["id"]
        return None
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
        profile = get_user_profile(user_id)
        
        if not profile:
            # Create new profile
            logger.info(f"Creating profile for {user_id} with data: {list(updates.keys())}")
            new_profile = {
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **updates
            }
            result = supabase.table("user_profiles").insert(new_profile).execute()
            
            if result.data:
                logger.info(f"✅ Profile created: {list(updates.keys())}")
                return True
            else:
                logger.error(f"Failed to create profile: {result}")
                return False
        else:
            # Update existing profile
            logger.info(f"Updating profile for {user_id} with data: {list(updates.keys())}")
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            result = supabase.table("user_profiles").update(updates).eq("user_id", user_id).execute()
            
            if result.data:
                logger.info(f"✅ Profile updated: {list(updates.keys())}")
                return True
            else:
                logger.error(f"Failed to update profile: {result}")
                return False
    except Exception as e:
        logger.error(f"Error in update_user_profile: {e}")
        return False

def save_to_memory(user_id: str, sender: str, message: str) -> bool:
    """Save message to conversation history"""
    try:
        result = supabase.table("conversations").insert({
            "user_id": user_id,
            "sender": sender,
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return bool(result.data)
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

def should_ask_onboarding_question(user_id: str) -> Optional[Dict]:
    """Check if user should be asked next onboarding question"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return ONBOARDING_QUESTIONS[0]
        
        # Check if personality selected first
        if not profile.get("personality_choice"):
            return None
        
        # Check if all questions answered
        questions_asked = profile.get("questions_asked_today", 0)
        if questions_asked < len(ONBOARDING_QUESTIONS):
            return ONBOARDING_QUESTIONS[questions_asked]
        return None
    except Exception as e:
        logger.error(f"Error checking onboarding: {e}")
        return None
    
#Helpful function to determine if we should ask post-onboarding questions (job, hobbies, etc.) 
def should_ask_post_onboarding_question(profile: Dict) -> bool:
    """Return True if it's time to ask the next post‑onboarding question."""
    if not profile.get("onboarding_complete"):
        return False
    
    asked = profile.get("post_onboarding_questions_asked", 0)
    if asked >= len(POST_ONBOARDING_QUESTIONS):
        return False  # all questions already asked
    
    last_asked = profile.get("last_post_onboarding_question_at")
    if not last_asked:
        return True  # never asked any, ask now
    
    # Convert to datetime (handle timezone)
    if isinstance(last_asked, str):
        last_asked = datetime.fromisoformat(last_asked)
    if last_asked.tzinfo is None:
        last_asked = last_asked.replace(tzinfo=timezone.utc)
    
    # Wait at least 24 hours between questions
    return (datetime.now(timezone.utc) - last_asked) > timedelta(hours=24)

# ===== INPUT VALIDATION FUNCTIONS =====
def clean_name(name_input: str) -> str:
    """Clean name input - remove prefixes and extra spaces"""
    # Remove common prefixes like "It's", "My", "I'm", "The"
    prefixes = ["it's ", "its ", "my ", "i'm ", "the ", "it is "]
    
    cleaned = name_input.strip()
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    logger.info(f"Cleaned name: '{name_input}' → '{cleaned}'")
    return cleaned

def validate_and_convert_age(age_input: str) -> Optional[str]:
    """Validate age input and convert text numbers to digits"""
    age_text = age_input.strip().lower()
    
    # Word to number mapping
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90'
    }
    
    # Handle hyphenated numbers like "twenty-one"
    if '-' in age_text:
        parts = age_text.split('-')
        if len(parts) == 2:
            tens = word_to_num.get(parts[0], None)
            ones = word_to_num.get(parts[1], None)
            if tens and ones:
                try:
                    num = int(tens) + int(ones)
                    logger.info(f"Converted age: '{age_input}' → '{num}'")
                    return str(num)
                except:
                    pass
    
    # Try to convert word to number
    if age_text in word_to_num:
        result = word_to_num[age_text]
        logger.info(f"Converted age: '{age_input}' → '{result}'")
        return result
    
    # Check if it's already a number
    try:
        age_num = int(age_text)
        # Validate reasonable age range (18-100)
        if 18 <= age_num <= 100:
            logger.info(f"Validated age: {age_num}")
            return str(age_num)
        else:
            logger.warning(f"You are not within the allowed age range: {age_num}")
            return None
    except ValueError:
        logger.warning(f"Invalid age input: '{age_input}'")
        return None

def validate_timezone(tz_input: str) -> Optional[str]:
    """Validate and normalize timezone input - NEW FUNCTION"""
    tz_input = tz_input.strip()
    
    # Try to match common abbreviations first
    if tz_input.lower() in COMMON_TIMEZONES:
        result = COMMON_TIMEZONES[tz_input.lower()]
        logger.info(f"Converted timezone: '{tz_input}' → '{result}'")
        return result
    
    # Try exact match in pytz
    if tz_input in VALID_TIMEZONES:
        logger.info(f"Validated timezone: '{tz_input}'")
        return tz_input
    
    # Try case-insensitive fuzzy match
    tz_lower = tz_input.lower()
    for valid_tz in VALID_TIMEZONES:
        if valid_tz.lower() == tz_lower:
            logger.info(f"Validated timezone (fuzzy): '{tz_input}' → '{valid_tz}'")
            return valid_tz
    
    logger.warning(f"Invalid timezone: '{tz_input}'")
    return None

def can_send_voice_today(user_id: str) -> bool:
    """Check if user can send voice today"""
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
    """Check if 6+ hours since last check-in - FIXED TIMEZONE BUG"""
    try:
        profile = get_user_profile(user_id)
        if not profile:
            return True
        
        last_checkin = profile.get("last_checkin_sent")
        if not last_checkin:
            return True
        
        # FIXED: Use timezone-aware datetime
        last_checkin_time = datetime.fromisoformat(last_checkin)
        if last_checkin_time.tzinfo is None:
            last_checkin_time = last_checkin_time.replace(tzinfo=timezone.utc)
        
        time_diff = datetime.now(timezone.utc) - last_checkin_time
        return time_diff.total_seconds() > 21600
    except Exception as e:
        logger.error(f"Error checking checkin: {e}")
        return True

def mark_checkin_sent(user_id: str) -> bool:
    """Mark that check-in was sent"""
    return update_user_profile(user_id, {"last_checkin_sent": datetime.now(timezone.utc).isoformat()})

def get_user_local_time(user_id: str) -> Optional[datetime]:
    """Get user's current local time based on their timezone - NEW FUNCTION"""
    try:
        profile = get_user_profile(user_id)
        if not profile or not profile.get("user_timezone"):
            return datetime.now(timezone.utc)  # Fallback to UTC
        
        tz_str = profile.get("user_timezone")
        try:
            user_tz = pytz.timezone(tz_str)
            return datetime.now(user_tz)
        except:
            logger.warning(f"Invalid timezone in profile: {tz_str}")
            return datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"Error getting user local time: {e}")
        return datetime.now(timezone.utc)

def is_intelligence_question(text: str) -> bool:
    """Detect if question is about intelligence/news/current affairs - NEW FUNCTION"""
    text_lower = text.lower()
    for keyword in INTELLIGENCE_KEYWORDS:
        if keyword in text_lower:
            logger.info(f"Detected intelligence question (keyword: '{keyword}')")
            return True
    return False


# =====================================================
# SECTION 3: CORE LOGIC (LLM, IMAGES, VOICE)
# =====================================================

def load_personality(personality_choice: str = 'female') -> str:
    """Load personality definition from file"""
    try:
        filename = PERSONALITY_FILES.get(personality_choice, 'soul_female.txt')
        filepath = os.path.join(SCRIPT_DIR, filename)
        
        logger.info(f"Loading: {personality_choice} ({filename})")
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filename}")
            fallback = os.path.join(SCRIPT_DIR, 'soul_female.txt')
            if os.path.exists(fallback):
                filepath = fallback
            else:
                return "You are Arya, a 24-year-old woman."
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"✅ Loaded {len(content)} chars")
            return content
    except Exception as e:
        logger.error(f"Error loading personality: {e}")
        return "You are Arya, a 24-year-old woman."

def get_random_error(error_type: str = "general") -> str:
    """Get random error message"""
    messages = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general"])
    return random.choice(messages)

def clean_response(text: str) -> str:
    """Remove thinking patterns from response"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        skip = False
        for pattern in THINKING_PATTERNS:
            if pattern in line.lower():
                skip = True
                break
        if len(line) > 200 and any(w in line.lower() for w in ["should", "needs", "arya", "first"]):
            skip = True
        if not skip and line.strip():
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else text.strip()

def should_generate_image(text: str) -> bool:
    """Check if response mentions image"""
    keywords = ["sends a photo", "sends an image", "sends a picture",
                "here's a photo", "here's an image", "here's a picture",
                "sending you a photo", "sending you an image", "*sends"]
    return any(kw in text.lower() for kw in keywords)

def split_into_messages(text: str, max_length: int = 1024) -> List[str]:
    """Split response into 2-3 messages"""
    if len(text) <= max_length:
        return [text]
    
    messages = []
    current = ""
    for paragraph in text.split('\n'):
        if len(current) + len(paragraph) + 1 > max_length:
            if current:
                messages.append(current.strip())
            current = paragraph
        else:
            current += ('\n' if current else '') + paragraph
    if current:
        messages.append(current.strip())
    return messages[:3]

def build_system_prompt_with_time(personality_content: str, user_id: str) -> str:
    """Build system prompt with current time context - NEW FUNCTION"""
    try:
        profile = get_user_profile(user_id)
        
        # Get user's local time
        user_local_time = get_user_local_time(user_id)
        day_name = user_local_time.strftime("%A")
        time_str = user_local_time.strftime("%I:%M %p")
        date_str = user_local_time.strftime("%B %d, %Y")
        hour = user_local_time.hour
        
        # Determine time of day
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        elif hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Build time context
        time_context = f"""
CURRENT DATE & TIME (User's Local Time):
- Date: {date_str} ({day_name})
- Time: {time_str}
- Time of Day: {time_of_day}

USE THIS CONTEXT to make your responses timely and relevant.
For example, if it's morning, you might say "good morning" naturally.
If it's late at night, you might notice they're up late.
Always be aware of what time it is for them right now.
"""
        
        return personality_content + time_context
    except Exception as e:
        logger.error(f"Error building system prompt: {e}")
        return personality_content

def generate_response(user_id: str, user_message: str, personality: str = 'female') -> Optional[str]:
    """Generate response using appropriate LLM based on question type - MODIFIED"""
    try:
        # Check if this is an intelligence question
        use_intelligence_brain = is_intelligence_question(user_message) and INTELLIGENCE_ENABLED
        
        if use_intelligence_brain:
            logger.info(f"Using INTELLIGENCE brain for: {user_message[:50]}...")
            return generate_intelligence_response(user_id, user_message)
        else:
            logger.info(f"Using ROLEPLAY brain ({personality}) for: {user_message[:50]}...")
            return generate_roleplay_response(user_id, user_message, personality)
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return None

def generate_roleplay_response(user_id: str, user_message: str, personality: str = 'female') -> Optional[str]:
    """Generate roleplay response using DeepSeek - EXTRACTED FUNCTION"""
    try:
        history = get_conversation_history(user_id, limit=10)
        soul_content = load_personality(personality)
        
        # Add time context to system prompt
        system_prompt = build_system_prompt_with_time(soul_content, user_id)
        
        # ===== Inject user profile data =====
        profile = get_user_profile(user_id)
        if profile:
            name = profile.get('user_name')
            age = profile.get('user_age')
            info_parts = []
            if name:
                info_parts.append(f"Name: {name}")
            if age:
                info_parts.append(f"Age: {age}")
            if info_parts:
                user_info = "User info: " + ", ".join(info_parts) + "\n\n"
                system_prompt = user_info + system_prompt
        
        messages = [{"role": "system", "content": system_prompt}]
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
        return clean_response(reply)
    except Exception as e:
        logger.error(f"Error in generate_roleplay_response: {e}")
        return None

def generate_intelligence_response(user_id: str, user_message: str) -> Optional[str]:
    """Generate intelligence/news response using Perplexity - NEW FUNCTION"""
    try:
        history = get_conversation_history(user_id, limit=5)
        
        # Build system prompt for intelligence
        system_prompt = f"""You are a knowledgeable and helpful AI assistant. The current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')}.

You have access to current information and news. Answer questions accurately and provide recent information when asked about current events, news, prices, weather, or any time-sensitive topics.

Be conversational and friendly, but factual and reliable. If you're not sure about something, say so.
Respond in a natural, human-like way."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history:
            role = "user" if msg["sender"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["message"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = intelligence_client.chat.completions.create(
            model=INTELLIGENCE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        logger.error(f"Error in generate_intelligence_response: {e}")
        return None

def generate_image_sync(prompt: str) -> Optional[BytesIO]:
    """Generate image using Eternal AI"""
    try:
        logger.info("Starting image generation")
        clean_prompt = prompt.replace("*", "").strip()
        enhanced_prompt = f"{clean_prompt} {IMAGE_MANDATORY_LOOK}"
        
        headers = {
            "x-api-key": ETERNAL_AI_API_KEY,
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": enhanced_prompt}]}],
            "agent": ETERNAL_AI_AGENT
        }
        
        submit_response = requests.post(ETERNAL_AI_SUBMIT_URL, json=payload, headers=headers, timeout=10)
        
        if submit_response.status_code != 200:
            logger.error(f"Submit failed: {submit_response.status_code}")
            return None
        
        submit_data = submit_response.json()
        request_id = submit_data.get("request_id")
        
        if not request_id:
            logger.error(f"No request_id: {submit_data}")
            return None
        
        for i in range(ETERNAL_AI_MAX_POLLS):
            response = requests.get(f"{ETERNAL_AI_POLL_URL}/{request_id}", headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                if status == "success":
                    result_url = data.get("result", {}).get("image_url")
                    if result_url:
                        img = requests.get(result_url, timeout=10)
                        if img.status_code == 200:
                            photo = BytesIO(img.content)
                            logger.info("✅ Image ready")
                            return photo
                    return None
                elif status == "failed":
                    logger.error(f"Generation failed: {data}")
                    return None
            
            __import__('time').sleep(min(0.5 + (i * 0.1), 2))
        
        logger.error("Polling timeout")
        return None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

def transcribe_voice_sync(voice_bytes: bytes) -> Optional[str]:
    """Transcribe voice"""
    try:
        response = transcription_client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=("audio.wav", voice_bytes),
            language="en"
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error transcribing: {e}")
        return None

def generate_voice_sync(text: str) -> Optional[BytesIO]:
    """Generate voice"""
    try:
        clean_text = re.sub(r'[^\w\s\?\!\.\,\-\'\"]', '', text)
        tts = gTTS(text=clean_text, lang='en', slow=False)
        voice_bytes = BytesIO()
        tts.write_to_fp(voice_bytes)
        voice_bytes.seek(0)
        return voice_bytes
    except Exception as e:
        logger.error(f"Error generating voice: {e}")
        return None


# =====================================================
# SECTION 4: MESSAGE HANDLERS (TELEGRAM)
# =====================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user_id_telegram = update.effective_user.id
    logger.info(f"User {user_id_telegram} started /start")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text("Sorry, couldn't set you up. Try again?")
        return
    
    profile = get_user_profile(user_id)
    if profile and profile.get("personality_choice"):
        name = PERSONALITIES_INFO[profile['personality_choice']]['name']
        await update.message.reply_text(f"hey! you already chose {name} 😊")
        return
    
    buttons = [
        [InlineKeyboardButton("👩 Arya (Female)", callback_data="select_personality_female")],
        [InlineKeyboardButton("👨 Alex (Male)", callback_data="select_personality_male")],
        [InlineKeyboardButton("🌸 Aris (Non-binary)", callback_data="select_personality_non-binary")]
    ]
    
    await update.message.reply_text(
        "hey! who would you like to talk to today? ✨",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

async def personality_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle personality selection"""
    query = update.callback_query
    user_id_telegram = update.effective_user.id
    
    personality_choice = query.data.replace("select_personality_", "")
    logger.info(f"User {user_id_telegram} selected: {personality_choice}")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await query.answer("Error, try /start again")
        return
    
    personality_info = PERSONALITIES_INFO.get(personality_choice)
    success = update_user_profile(user_id, {
        "personality_choice": personality_choice,
        "questions_asked_today": 0,
        "onboarding_complete": False
    })
    
    if success:
        await query.answer()
        await query.edit_message_text(text=f"awesome, i'm {personality_info['name']}! 💫")
        
        await asyncio.sleep(0.5)
        first_q = ONBOARDING_QUESTIONS[0]
        await context.bot.send_message(chat_id=user_id_telegram, text=first_q["question"])
    else:
        await query.answer("Failed to save choice, try /start again")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    user_message = update.message.text
    
    logger.info(f"Message from {user_id_telegram}: {user_message[:50]}...")
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error("general"))
        return
    
    profile = get_user_profile(user_id)
    
    # Check personality selected
    if not profile or not profile.get("personality_choice"):
        await update.message.reply_text("you need to pick a personality first! use /start to choose 😊")
        return
    
    # Check onboarding
    onboarding_q = should_ask_onboarding_question(user_id)
    
    if onboarding_q:
        questions_asked = profile.get("questions_asked_today", 0)
        logger.info(f"Onboarding Q#{questions_asked + 1}: {onboarding_q['field']}")
        
        if questions_asked == 0:
            # ===== QUESTION 1: NAME =====
            # Clean the name input
            cleaned_name = clean_name(user_message)
            
            if not cleaned_name or len(cleaned_name) < 2:
                await update.message.reply_text("hmm that doesn't look like a name 😅 try again?")
                logger.warning(f"Invalid name input: '{user_message}'")
                return
            
            logger.info(f"✅ Saving name: '{cleaned_name}'")
            update_user_profile(user_id, {
                "user_name": cleaned_name,
                "questions_asked_today": 1
            })
            await asyncio.sleep(0.3)
            
            # ASK NEXT QUESTION (Age)
            next_q = ONBOARDING_QUESTIONS[1]
            await update.message.reply_text(next_q["question"])
            logger.info(f"Asked Q2: {next_q['question'][:50]}...")
            return
        
        elif questions_asked == 1:
            # ===== QUESTION 2: AGE =====
            # Validate and convert age
            validated_age = validate_and_convert_age(user_message)
            
            if not validated_age:
                await update.message.reply_text("that doesn't look like a valid age 😅 please tell me a number (like 25 or 'twenty-five')")
                logger.warning(f"Invalid age input: '{user_message}'")
                return
            
            logger.info(f"✅ Saving age: '{validated_age}'")
            update_user_profile(user_id, {
                "user_age": validated_age,
                "questions_asked_today": 2
            })
            
            await asyncio.sleep(0.3)
            
            # ASK NEXT QUESTION (Timezone) - NEW
            next_q = ONBOARDING_QUESTIONS[2]
            await update.message.reply_text(next_q["question"])
            logger.info(f"Asked Q3: {next_q['question'][:50]}...")
            return
        
        elif questions_asked == 2:
            # ===== QUESTION 3: TIMEZONE - NEW =====
            # Validate timezone
            validated_tz = validate_timezone(user_message)
            
            if not validated_tz:
                await update.message.reply_text("i'm not sure that's a valid timezone 😅 try 'America/New_York', 'Europe/London', 'Asia/Tokyo', etc")
                logger.warning(f"Invalid timezone input: '{user_message}'")
                return
            
            logger.info(f"✅ Saving timezone: '{validated_tz}'")
            update_user_profile(user_id, {
                "user_timezone": validated_tz,
                "questions_asked_today": 3,
                "onboarding_complete": True
            })
            
            await update.message.reply_text(f"Nice to meet you, {profile.get('user_name', 'there')}! 😊 Now we can chat whenever, and I'll know what time it is for you 🌍")
            logger.info(f"✅ ONBOARDING COMPLETE!")
            await asyncio.sleep(0.5)
            return
    
    # Normal conversation
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
        
        if should_generate_image(arya_reply):
            msgs = ["Wait, let me find that photo for you. 📸", "Let me click the best one for you... ✨",
                    "Hold on, let me get a good shot... 📷", "One sec, let me find the perfect photo... 💫"]
            await update.message.reply_text(random.choice(msgs))
            await send_image_task(context, chat_id, arya_reply)
        else:
            for msg in split_into_messages(arya_reply):
                await update.message.reply_text(msg)
                await asyncio.sleep(0.3)

        # ===== POST-ONBOARDING HANDLING =====
        # After sending the normal reply, handle any pending questions or ask new ones
        if profile.get("onboarding_complete"):
            # Check if we have a pending question
            pending_field = profile.get("pending_question_field")
            if pending_field:
                answer = user_message.strip()
                if answer:
                    # Save answer to the corresponding column
                    update_user_profile(user_id, {pending_field: answer})
                    # Clear pending state
                    update_user_profile(user_id, {
                        "pending_question_field": None,
                        "pending_question_asked_at": None
                    })
                    logger.info(f"Saved answer for {pending_field}: {answer[:50]}...")
                    # Refresh profile after update (optional, but safe)
                    profile = get_user_profile(user_id)
            
            # Now see if we should ask the next question (using possibly refreshed profile)
            if should_ask_post_onboarding_question(profile):
                asked = profile.get("post_onboarding_questions_asked", 0)
                next_q = POST_ONBOARDING_QUESTIONS[asked]
                # Send the question
                await update.message.reply_text(next_q["question"])
                # Update tracking: increment counter, set pending, record time
                now_iso = datetime.now(timezone.utc).isoformat()
                update_user_profile(user_id, {
                    "post_onboarding_questions_asked": asked + 1,
                    "last_post_onboarding_question_at": now_iso,
                    "pending_question_field": next_q["field"],
                    "pending_question_asked_at": now_iso
                })
                logger.info(f"Asked post‑onboarding question #{asked+1}: {next_q['field']}")
        # =====================================
    
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(get_random_error("general"))

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    user_id_telegram = update.effective_user.id
    chat_id = update.effective_chat.id
    
    user_id = get_or_create_user(user_id_telegram)
    if not user_id:
        await update.message.reply_text(get_random_error("voice"))
        return
    
    profile = get_user_profile(user_id)
    if not profile or not profile.get("personality_choice"):
        await update.message.reply_text("use /start to pick a personality first! 😊")
        return
    
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()
        
        transcribed = await asyncio.to_thread(transcribe_voice_sync, bytes(voice_bytes))
        if not transcribed:
            await update.message.reply_text(get_random_error("voice"))
            return
        
        save_to_memory(user_id, "user", f"[voice] {transcribed}")
        update_user_profile(user_id, {"last_message_from_user": datetime.now(timezone.utc).isoformat()})
        
        personality = profile.get("personality_choice", "female")
        reply = generate_response(user_id, transcribed, personality)
        
        if not reply:
            await update.message.reply_text(get_random_error("general"))
            return
        
        save_to_memory(user_id, "arya", reply)
        
        for msg in split_into_messages(reply):
            await update.message.reply_text(msg)
            await asyncio.sleep(0.3)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(get_random_error("voice"))

async def send_image_task(context: CallbackContext, chat_id: int, prompt: str):
    """Generate and send image"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        
        clean = prompt.replace("*", "").strip()[:300]
        photo = await asyncio.to_thread(generate_image_sync, clean)
        
        if photo:
            await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="for you...")
            logger.info("✅ Photo sent")
        else:
            await context.bot.send_message(chat_id=chat_id, text=get_random_error("image"))
    except Exception as e:
        logger.error(f"Error in image task: {e}")

async def send_voice_task(context: CallbackContext, chat_id: int, text: str, user_id: str = None):
    """Generate and send voice"""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
        
        voice_text = text[:500]
        voice = await asyncio.to_thread(generate_voice_sync, voice_text)
        
        if voice:
            await context.bot.send_voice(chat_id=chat_id, voice=voice)
            if user_id:
                mark_voice_sent(user_id)
    except Exception as e:
        logger.error(f"Error in voice task: {e}")


# =====================================================
# SECTION 5: BACKGROUND JOBS & MAIN
# =====================================================

async def check_for_checkins(context: CallbackContext):
    """Background job: Send check-ins"""
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
                except Exception as e:
                    logger.error(f"Checkin failed: {e}")
    except Exception as e:
        logger.error(f"Checkin job error: {e}")

def main():
    """Start the bot"""
    logger.info("="*70)
    logger.info("🚀 ARYA 4.0 - FINAL PRODUCTION VERSION")
    logger.info("="*70)
    logger.info(f"Primary Brain (Roleplay): {BRAIN_MODEL}")
    logger.info(f"Secondary Brain (Intelligence): {INTELLIGENCE_MODEL}")
    logger.info(f"Voice In: {TRANSCRIPTION_PROVIDER}")
    logger.info(f"Voice Out: {TTS_PROVIDER}")
    logger.info(f"Images: {IMAGE_PROVIDER}")
    logger.info("="*70)
    logger.info("Personalities: Arya (Female), Alex (Male), Aris (Non-binary)")
    logger.info("Mandatory Data: Name, Age, Timezone")
    logger.info("Features: TIME-AWARE messages, DUAL-BRAIN intelligence system")
    logger.info("Security: RLS Enabled, Data Private, Timezone-aware")
    logger.info("="*70)
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(personality_selection_callback, pattern="^select_personality_"))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    app.job_queue.run_repeating(check_for_checkins, interval=25200, first=10)
    
    logger.info("✅ All handlers registered!")
    logger.info("✅ /start command ready")
    logger.info("✅ Personality selection ready")
    logger.info("✅ Data collection ready (Name, Age, Timezone)")
    logger.info("✅ Time-aware responses enabled")
    logger.info("✅ Dual-brain intelligence routing enabled")
    logger.info("✅ RLS enabled (secure)")
    logger.info("💬 BOT IS RUNNING")
    logger.info("="*70)
    
    app.run_polling()

if __name__ == '__main__':
    main()