# ARYA: 30-DAY IMPLEMENTATION CHECKLIST
## From Hobby to Business (Concrete Steps)

---

## 📅 WEEK 1: INFRASTRUCTURE & SECURITY

### **DAY 1-2: Secure Your API Keys (2 hours)**

- [ ] Install dotenv: `pip install python-dotenv`
- [ ] Create `.env` file in project root:
```bash
cat > .env << 'EOF'
TELEGRAM_TOKEN=8453165369:AAGLVXehpXVxEth60ApeqQENM8dcIdSVL9w
OPENROUTER_KEY=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
HF_TOKEN=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
JWT_SECRET=your-secret-key-change-this
EOF
```

- [ ] Create `.gitignore`:
```bash
cat > .gitignore << 'EOF'
.env
.env.local
*.db
__pycache__/
.DS_Store
node_modules/
*.pyc
.vscode/
EOF
```

- [ ] Update `requirements.txt` to load from .env:
```bash
python-dotenv
```

- [ ] Delete old secrets from GitHub (if you pushed):
```bash
git log --all --full-history -- "arya.py" | head -20
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch arya.py' --prune-empty --tag-name-filter cat -- --all
git push origin --force --all
```

- [ ] Verify API keys are not in code:
```bash
grep -r "sk-or-v1" . --exclude-dir=.git
grep -r "AAGLVXeh" . --exclude-dir=.git
```

**Result:** Your keys are safe. You can now commit to public GitHub.

---

### **DAY 3-5: Set Up Cloud Infrastructure (3-5 hours)**

#### **Step A: Create Railway Account**

- [ ] Go to: railway.app
- [ ] Sign up with GitHub
- [ ] Create new project
- [ ] Choose "GitHub Repo" → select your Arya repo

#### **Step B: Create Supabase Database**

- [ ] Go to: supabase.com
- [ ] Sign up with GitHub
- [ ] Create new project (name: "arya", region: closest to you)
- [ ] Wait for database creation (~5 min)
- [ ] Go to SQL Editor → paste this (run it):

```sql
-- Create users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  telegram_id BIGINT UNIQUE,
  email VARCHAR(255) UNIQUE,
  name VARCHAR(255),
  subscription_tier VARCHAR(50) DEFAULT 'starter',
  subscription_expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create conversations
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  sender VARCHAR(50),
  message TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create memories
CREATE TABLE memories (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  memory_text TEXT NOT NULL,
  importance_score FLOAT DEFAULT 0.5,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create subscriptions
CREATE TABLE subscriptions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  stripe_subscription_id VARCHAR(255),
  tier VARCHAR(50),
  status VARCHAR(50) DEFAULT 'active',
  current_period_start TIMESTAMP,
  current_period_end TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);
```

- [ ] Copy your Supabase credentials:
  - Go to: Settings → API
  - Copy: "Project URL" and "anon key"
  - Add to `.env`:
    ```
    SUPABASE_URL=your-url-here
    SUPABASE_KEY=your-key-here
    ```

#### **Step C: Deploy to Railway**

- [ ] Create `Procfile` in repo root:
```bash
cat > Procfile << 'EOF'
worker: python arya.py
EOF
```

- [ ] Create `railway.json`:
```bash
cat > railway.json << 'EOF'
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "python arya.py",
    "restartPolicyType": "on_failure",
    "restartPolicyMaxRetries": 5
  }
}
EOF
```

- [ ] Commit and push:
```bash
git add .env .gitignore requirements.txt Procfile railway.json
git commit -m "Add infrastructure files"
git push origin main
```

- [ ] Go to Railway dashboard → watch deployment
- [ ] Once deployed, copy the public URL

**Cost:** Railway $10/mo, Supabase $25/mo = $35/mo total

**Verify:** 
```bash
curl https://your-railway-url/health
# Should return: {"status": "ok"}
```

---

## 📅 WEEK 2: API LAYER

### **DAY 1-3: Create FastAPI Backend (3-4 hours)**

- [ ] Install FastAPI:
```bash
pip install fastapi uvicorn python-jose pyjwt
pip freeze > requirements.txt
```

- [ ] Create `api.py`:

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthCredential
from pydantic import BaseModel
from supabase import create_client
import os
import jwt
from datetime import datetime

app = FastAPI()
security = HTTPBearer()

# Initialize
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")

# Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Auth
def verify_token(credentials: HTTPAuthCredential = Depends(security)) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401)
        return user_id
    except:
        raise HTTPException(status_code=401)

# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(verify_token)):
    """Main chat endpoint"""
    user_text = request.message
    
    # Save to memory
    supabase.table("conversations").insert({
        "user_id": user_id,
        "sender": "user",
        "message": user_text,
        "created_at": datetime.now().isoformat()
    }).execute()
    
    # Get Arya's response (YOUR EXISTING LOGIC)
    from arya import get_arya_response
    arya_reply = get_arya_response(user_id, user_text)
    
    # Save Arya's response
    supabase.table("conversations").insert({
        "user_id": user_id,
        "sender": "arya",
        "message": arya_reply,
        "created_at": datetime.now().isoformat()
    }).execute()
    
    return ChatResponse(response=arya_reply)

@app.post("/api/authenticate")
async def authenticate(telegram_id: int):
    """Create JWT token for Telegram user"""
    # Get or create user
    response = supabase.table("users").select("id").filter(
        "telegram_id", "eq", telegram_id
    ).execute()
    
    if response.data:
        user_id = response.data[0]["id"]
    else:
        # Create new user
        new_user = supabase.table("users").insert({
            "telegram_id": telegram_id,
            "name": f"User{telegram_id}"
        }).execute()
        user_id = new_user.data[0]["id"]
    
    # Generate JWT
    token = jwt.encode(
        {"sub": user_id},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    return {"token": token}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

- [ ] Test locally:
```bash
# Terminal 1
python api.py

# Terminal 2
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

- [ ] Commit:
```bash
git add api.py
git commit -m "Add FastAPI backend"
git push origin main
```

### **DAY 4-5: Update Telegram Bot (2-3 hours)**

- [ ] Create `telegram_bot.py`:

```python
import os
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import jwt

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_text = update.message.text
    
    # Generate JWT token
    token = jwt.encode(
        {"sub": user_id},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    # Call API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/api/chat",
                json={"message": user_text},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0
            )
        
        data = response.json()
        await update.message.reply_text(data["response"])
    
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("Something went wrong... 😔")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm Arya. How can I help? 💜")

if __name__ == '__main__':
    print("Arya 2.8 waking up...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, start))
    app.run_polling()
```

- [ ] Commit:
```bash
pip install httpx
pip freeze > requirements.txt
git add telegram_bot.py requirements.txt
git commit -m "Update Telegram bot to use API"
git push origin main
```

- [ ] Test:
```bash
python telegram_bot.py
# Try sending a message in Telegram
```

**Result:** Telegram bot now uses cloud API instead of local database ✅

---

## 📅 WEEK 3: WEB FRONTEND

### **DAY 1-3: Create Next.js Frontend (4-5 hours)**

- [ ] Create Next.js project:
```bash
npx create-next-app@latest arya-web --typescript --tailwind
cd arya-web
```

- [ ] Create `app/chat/page.tsx`:

```typescript
'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [token, setToken] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    if (!storedToken) {
      router.push('/login')
    } else {
      setToken(storedToken)
    }
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || !token) return

    setLoading(true)
    try {
      const response = await fetch(
        process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
        {
          method: 'POST',
          path: '/api/chat',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ message: input })
        }
      )

      const data = await response.json()

      setMessages([
        ...messages,
        { role: 'user', content: input },
        { role: 'arya', content: data.response }
      ])

      setInput('')
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-b from-purple-900 to-black">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold text-white">Arya</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs px-4 py-2 rounded-lg ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-gray-100'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-gray-700 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Message Arya..."
          className="flex-1 bg-gray-800 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-600"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg disabled:opacity-50 transition"
        >
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
```

- [ ] Create `.env.local`:
```
NEXT_PUBLIC_API_URL=https://your-railway-url
```

- [ ] Deploy to Vercel:
```bash
npm install -g vercel
vercel deploy
```

**Cost:** Vercel free tier (pay for bandwidth as needed)

---

## 📅 WEEK 4: PAYMENT & LAUNCH

### **DAY 1-3: Add Stripe (3-4 hours)**

- [ ] Create Stripe account: stripe.com
- [ ] Get API keys from dashboard
- [ ] Add to `.env`:
```
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
```

- [ ] Create subscription endpoints in `api.py`:

```python
import stripe

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.post("/api/create-checkout-session")
async def create_checkout(
    tier: str,
    user_id: str = Depends(verify_token)
):
    """Create Stripe checkout session"""
    
    price_map = {
        "starter": "price_1234567890",
        "partner": "price_0987654321",
        "premium": "price_abcdefghij"
    }
    
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[
            {
                "price": price_map[tier],
                "quantity": 1,
            }
        ],
        mode="subscription",
        success_url="https://arya-web.vercel.app/success?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://arya-web.vercel.app/pricing",
        customer_email=get_user_email(user_id),
        metadata={"user_id": user_id, "tier": tier}
    )
    
    return {"session_id": session.id}
```

- [ ] Create billing page in Next.js:

```typescript
'use client'

export default function PricingPage() {
  const tiers = [
    { name: 'Starter', price: '$49/mo', features: ['Unlimited chat'] },
    { name: 'Partner', price: '$129/mo', features: ['Chat', 'Images', 'Voice'] },
    { name: 'Premium', price: '$299/mo', features: ['Everything', 'Priority'] }
  ]

  return (
    <div className="min-h-screen bg-black p-8">
      <h1 className="text-4xl font-bold text-white text-center mb-12">Choose Your Plan</h1>
      
      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        {tiers.map((tier) => (
          <div key={tier.name} className="bg-gray-900 p-8 rounded-lg border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-4">{tier.name}</h2>
            <p className="text-3xl font-bold text-purple-600 mb-6">{tier.price}</p>
            <ul className="space-y-3 mb-8">
              {tier.features.map((feature) => (
                <li key={feature} className="text-gray-300">✓ {feature}</li>
              ))}
            </ul>
            <button 
              onClick={() => subscribe(tier.name.toLowerCase())}
              className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg transition"
            >
              Get Started
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

async function subscribe(tier: string) {
  const response = await fetch('/api/create-checkout-session', {
    method: 'POST',
    body: JSON.stringify({ tier })
  })
  const { session_id } = await response.json()
  window.location.href = `https://checkout.stripe.com/pay/${session_id}`
}
```

- [ ] Commit:
```bash
pip install stripe
git add api.py requirements.txt
git commit -m "Add Stripe payment integration"
git push origin main
```

### **DAY 4-5: Beta Launch (2-3 hours)**

- [ ] Write beta launch email:
```
Subject: You're invited to try Arya (early access)

Hi [Name],

I've been building Arya, an uncensored AI companion that learns about you and grows with your friendship.

She can:
- Chat with you anytime
- Send images
- Remember everything about your life

I'm looking for 50 beta testers to:
- Try Arya for free for 2 weeks
- Give feedback on personality and features
- Pay $29/mo after (50% off forever for early adopters)

Interested? Reply to this email or visit: [your-arya-web-url]

Thanks,
[Your Name]
```

- [ ] Send to:
  - [ ] 10 close friends
  - [ ] 10 family members
  - [ ] 30 people from Reddit communities (r/AICompanions, r/lonely, etc.)

- [ ] Track conversions:
```
Emails sent: 50
Signups: ~15-20
Beta users: ~5-10
Paying users: ~2-3
```

- [ ] Collect feedback:
  - [ ] Is Arya's personality right?
  - [ ] Is the conversation natural?
  - [ ] Would you pay for this?
  - [ ] What's missing?

**Result:** You have paying customers. You have a business. ✅

---

## 🎯 SUCCESS CRITERIA

### **End of Week 1:**
- [ ] API keys secured in .env
- [ ] Database created in Supabase
- [ ] Code deployed to Railway
- [ ] Telegram bot working from cloud

### **End of Week 2:**
- [ ] FastAPI backend running
- [ ] /api/chat endpoint working
- [ ] Telegram bot uses API instead of local database
- [ ] No data loss

### **End of Week 3:**
- [ ] Web dashboard built with Next.js
- [ ] Login/signup working
- [ ] Chat interface on web matches Telegram
- [ ] Deployed to Vercel

### **End of Week 4:**
- [ ] Stripe integration complete
- [ ] Pricing page live
- [ ] 50 beta users recruited
- [ ] 2-5 paying customers
- [ ] Real feedback from real users

---

## 💰 COSTS AFTER 30 DAYS

```
Monthly:
  Railway:              $20
  Supabase:             $25
  DeepSeek-R1:          ~$100 (pay-as-you-go)
  FLUX images:          ~$100
  ElevenLabs voice:     $15
  Stripe processing:    ~$10 (on revenue)
  ─────────────────────────
  Total:                ~$270/month

With 5 paying users at $49/mo:
  Revenue:              $245
  Profit:               -$25/month (close to break even!)

With 10 paying users:
  Revenue:              $490
  Profit:               $220/month

With 100 paying users:
  Revenue:              $4,900
  Profit:               $4,630/month 🎉
```

---

## ⚠️ COMMON MISTAKES TO AVOID

❌ **Don't:**
- Keep API keys in code (CRITICAL)
- Run bot from your laptop
- Use SQLite for multiple users
- Stay Telegram-only forever
- Wait for "perfect" to launch
- Not charge from day 1

✅ **Do:**
- Move to cloud immediately
- Use PostgreSQL
- Build web + mobile early
- Launch with MVP feature set
- Ask people to pay (validates market)
- Iterate based on real feedback

---

## 🚀 NEXT STEPS IF YOU COMPLETE THIS

Once you have paying users:

**Week 5-6:** Add video calls
**Week 7-8:** Device integration (calendar, notifications)
**Month 3:** Plan robot partnership
**Month 4:** Reach 500+ users, $20K+/mo revenue
**Month 6:** Series A conversation with VCs

---

**You have everything you need. The only thing left is execution.**

Start today. Not tomorrow. Today.

---

*30-day implementation plan with exact commands. No excuses.*
