# app.py
# MindEase: Mood Journal + AI Mood Analyzer + Mood History + Self-care Tools + Crisis Support

import os
from datetime import datetime, date

import pandas as pd
import streamlit as st
import google.generativeai as genai

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="MindEase", layout="centered", page_icon="🌱")

DATA_FILE = "mood_log.csv"

# Gemini API Key
GEMINI_API_KEY = "AIzaSyB5d71HayaWgpFqSGnOqWX5JkJpgK11kmQ"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

MOODS = [
    "Happy", "Calm", "Okay", "Worried", "Anxious", "Stressed",
    "Sad", "Angry", "Frustrated", "Tired"
]

# For charting: higher = better
MOOD_SCORE = {
    "Happy": 5,
    "Calm": 4,
    "Okay": 3,
    "Worried": 2,
    "Anxious": 2,
    "Stressed": 1,
    "Sad": 1,
    "Angry": 1,
    "Frustrated": 1,
    "Tired": 2,
}

# ---------- EMOTION MAPPING FOR AI DETECTION ----------
EMOTION_DATA = {
    # Positive emotions
    "Happy": {"emoji": "😊", "color": "success", "category": "Positive", "score": 5},
    "Excited": {"emoji": "🎉", "color": "success", "category": "Positive", "score": 5},
    "Calm": {"emoji": "😌", "color": "success", "category": "Positive", "score": 4},
    "Grateful": {"emoji": "🙏", "color": "success", "category": "Positive", "score": 4},
    "Hopeful": {"emoji": "✨", "color": "success", "category": "Positive", "score": 4},
    
    # Anxious/Stressed
    "Anxious": {"emoji": "😰", "color": "warning", "category": "Negative", "score": 2},
    "Stressed": {"emoji": "😫", "color": "warning", "category": "Negative", "score": 2},
    "Overwhelmed": {"emoji": "😩", "color": "warning", "category": "Negative", "score": 1},
    "Scared": {"emoji": "😨", "color": "warning", "category": "Negative", "score": 1},
    
    # Sad/Lonely
    "Sad": {"emoji": "😢", "color": "error", "category": "Negative", "score": 1},
    "Lonely": {"emoji": "💔", "color": "error", "category": "Negative", "score": 1},
    "Empty": {"emoji": "🕳️", "color": "error", "category": "Negative", "score": 1},
    "Tired": {"emoji": "😴", "color": "info", "category": "Neutral", "score": 2},
    
    # Angry/Frustrated
    "Angry": {"emoji": "😠", "color": "error", "category": "Negative", "score": 1},
    "Frustrated": {"emoji": "😤", "color": "error", "category": "Negative", "score": 1},
    
    # Guilty/Ashamed
    "Guilty": {"emoji": "😞", "color": "error", "category": "Negative", "score": 1},
    "Ashamed": {"emoji": "😶", "color": "error", "category": "Negative", "score": 1},
    
    # Neutral/Fallback
    "Okay": {"emoji": "😐", "color": "info", "category": "Neutral", "score": 3},
    "Neutral": {"emoji": "😐", "color": "info", "category": "Neutral", "score": 3},
    "Worried": {"emoji": "😟", "color": "warning", "category": "Negative", "score": 2}
}

# ---------- CRISIS KEYWORDS  ----------
CRISIS_KEYWORDS = [
    # Self-harm methods
    "kill myself", "suicide", "want to die", "end my life", 
    "hurt myself", "stab myself", "cut myself", "slash",
    "hang myself", "jump off", "overdose", "shoot myself",
    "self-harm", "harm myself", "take my life", "end it all",
    "die", "dead", "kill", "slit", "wrist", "bleed",
    
    # Harm to others
    "kill someone", "hurt someone", "stab someone", "shoot someone",
    "hurt them", "kill them", "attack", "violent",
    
    # Distress indicators
    "hopeless", "worthless", "no reason to live",
    "can't go on", "give up", "better off dead"
]

# ---------- HELPER FUNCTIONS ----------
def load_log() -> pd.DataFrame:
    """Load mood journal from CSV"""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            for col in ["date", "mood", "note", "score", "source"]:
                if col not in df.columns:
                    df[col] = "" if col != "score" else 3
            return df
        except Exception:
            return pd.DataFrame(columns=["date", "mood", "note", "score", "source"])
    return pd.DataFrame(columns=["date", "mood", "note", "score", "source"])

def save_entry(mood: str, note: str, source: str = "ai"):
    """Save mood entry to CSV with today's date"""
    entry_date = date.today().isoformat()
    df = load_log()
    
    # Get score based on emotion
    score = EMOTION_DATA.get(mood, {}).get("score", 3)
    
    new_row = {
        "date": entry_date,
        "mood": mood,
        "note": note,
        "score": score,
        "source": source
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def analyze_with_gemini(journal_text: str):
    """Send text to Gemini AI for emotion analysis"""
    prompt_text = f"""
    You are a highly empathetic mental health assistant and mood analyzer.
    
    Read this journal entry and identify the **specific, real emotion** the person is feeling.
    
    Journal Entry: \"\"\"{journal_text}\"\"\"
    
    **YOUR TASK:**
    1. Identify the single most dominant emotion from this list: 
       Happy, Excited, Calm, Grateful, Hopeful, Anxious, Stressed, Overwhelmed, 
       Sad, Lonely, Angry, Frustrated, Tired, Empty, Scared, Ashamed, Guilty, Worried, Okay, Neutral
    
    2. Write ONE short, warm sentence explaining why you think they feel this way.
    
    **FORMAT (exactly like this):**
    Emotion: [one word from list above]
    [Your empathetic explanation]
    
    Example:
    Emotion: Anxious
    You seem worried about things that haven't happened yet.
    
    Be specific. Don't just say "Positive" or "Negative" - find the real emotion.
    """
    
    try:
        response = GEMINI_MODEL.generate_content(prompt_text)
        result = response.text.strip()
        
        # Extract emotion
        lines = result.split('\n')
        emotion = "Neutral"
        explanation = result
        
        for line in lines:
            if line.lower().startswith("emotion:"):
                emotion = line.replace("Emotion:", "").replace("emotion:", "").strip()
                break
                
        return emotion, explanation
    except Exception as e:
        return "Neutral", f"AI analysis unavailable. Your entry was saved."

def check_crisis_keywords(text: str) -> bool:
    """Check if text contains crisis keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

# ---------- UI COMPONENTS (CRISIS SUPPORT) ----------
def display_crisis_support():
    """Display Singapore-specific crisis support resources"""
    st.error("""
        🚨 **CRITICAL ALERT - IMMEDIATE HELP NEEDED** 🚨
        
        Your journal entry contains thoughts about harming yourself or others.
        
        **IN SINGAPORE, HELP IS AVAILABLE 24/7:**
        
        📞 **Samaritans of Singapore (SOS):** 1767
        💬 **SOS Care Text:** 9151 1767 (WhatsApp)
        🌐 **Chat Online:** https://www.sos.org.sg/
        
        🏥 **IMH Mental Health Helpline:** 6389 2222
        👥 **TOUCHline (Counselling):** 6804 6555
        🧑‍⚕️ **CHAT (Youth):** 6493 6500 / www.chat.mentalhealth.sg
        
        ---
        **Please reach out now. You are not alone. 💛**
        """)

def display_safety_block():
    """Display general safety resources"""
    st.markdown("### 🚨 Need Immediate Help? (Singapore)")
    st.markdown("""
- **Samaritans of Singapore (SOS)**: **1767** (24/7)
- **Institute of Mental Health (IMH) Mental Health Helpline**: **6389 2222** (24/7)
- **Singapore Association for Mental Health (SAMH)**: **1800 283 7019**
- **Singapore Children's Society – Tinkle Friend**: **1800 274 4788** (for children & youth)

If you feel unsafe right now, please call one of the numbers above or reach out to a trusted adult immediately.
""")

def display_emotion_result(emotion: str, explanation: str):
    """Display emotion with proper emoji and color"""
    if emotion in EMOTION_DATA:
        emoji = EMOTION_DATA[emotion]["emoji"]
        color = EMOTION_DATA[emotion]["color"]
    else:
        emoji = "🧠"
        color = "info"
    
    if color == "success":
        st.success(f"{emoji} **{emotion}**\n\n{explanation}")
    elif color == "warning":
        st.warning(f"{emoji} **{emotion}**\n\n{explanation}")
    elif color == "error":
        st.error(f"{emoji} **{emotion}**\n\n{explanation}")
    else:
        st.info(f"{emoji} **{emotion}**\n\n{explanation}")

# ---------- IMPROVED SELF-CARE FUNCTIONS ----------
def get_self_care_suggestions(emotion: str):
    """Return detailed, actionable self-care suggestions based on emotion"""
    
    suggestions = {
        # ----------------------------------------
        # ANXIETY & STRESS
        # ----------------------------------------
        "Anxious": {
            "title": "🌀 Calm Your Nervous System",
            "color": "warning",
            "quick_relief": [
                "🌬️ **4-7-8 Breathing:** Inhale 4 sec → Hold 7 sec → Exhale 8 sec. Repeat 4 times.",
                "❄️ **Cold Water:** Splash cold water on your wrists and face. This triggers the 'mammalian dive reflex' and slows your heart rate.",
                "🖐️ **Palm Pressure:** Press your thumb firmly into your opposite palm for 30 seconds.",
                "🌳 **5-4-3-2-1 Grounding:** Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste."
            ],
            "mental_shifts": [
                "🧠 **Name It to Tame It:** Say out loud: 'I am feeling anxious right now. This is just a feeling, and it will pass.'",
                "📅 **Worry Time:** Postpone your worry. Tell yourself: 'I will worry about this at 7pm for 15 minutes. Not now.'",
                "🎯 **Control Check:** What's ONE small thing you can control right now? Do that thing."
            ],
            "physical": [
                "🚶 **Walk for 5 minutes** without your phone. Feel your feet on the ground.",
                "🙆 **Shoulder rolls:** Roll your shoulders backward 10 times. We hold anxiety in our shoulders.",
                "💧 **Drink water:** Dehydration mimics anxiety symptoms. Drink a full glass slowly."
            ],
            "quote": "Anxiety is a wave. You can't stop it, but you can learn to surf.",
            "affirmation": "I am safe in this moment. This feeling is temporary."
        },
        
        "Stressed": {
            "title": "📉 Lower Your Stress Level",
            "color": "warning",
            "quick_relief": [
                "✋ **STOP Technique:** Stop what you're doing, Take a breath, Observe how you feel, Proceed with kindness.",
                "⏸️ **The 2-Minute Rule:** If something takes less than 2 minutes, do it now. If not, write it down for later.",
                "📋 **Brain Dump:** Write down everything in your head. Don't organize, just dump. Close the notebook."
            ],
            "mental_shifts": [
                "🎯 **One Thing:** What's the ONE thing that matters most right now? Do just that.",
                "🚫 **Permission to say no:** You are allowed to protect your energy. Not everything is urgent.",
                "🌟 **Progress > Perfection:** Done is better than perfect. Lower the bar today."
            ],
            "physical": [
                "☕ **Warm drink:** Hold a warm cup of tea. Focus on the warmth in your hands.",
                "🧘 **Neck stretch:** Slowly tilt your ear toward your shoulder. Hold 30 seconds each side.",
                "👣 **Barefoot:** Take your socks off and press your feet into the floor."
            ],
            "quote": "You don't have to see the whole staircase. Just take the first step.",
            "affirmation": "I am capable. I can handle this one step at a time."
        },
        
        "Overwhelmed": {
            "title": "🌊 Reduce Overwhelm",
            "color": "warning",
            "quick_relief": [
                "🛑 **Emergency Pause:** Stop everything. Set a timer for 5 minutes. Do nothing but breathe.",
                "📦 **The Box Method:** Imagine putting your worries in a box and closing the lid. You can open it later.",
                "🎯 **Micro-Step:** What's the TINIEST thing you can do? Open a document. Pick up one sock. Start there."
            ],
            "mental_shifts": [
                "🧹 **Single-task:** Multitasking is a myth. Do ONE thing at a time.",
                "🗣️ **Ask for help:** 'I'm overwhelmed, can you help me with [specific task]?'",
                "📉 **Lower expectations:** Today, 'good enough' is enough. You can do more tomorrow."
            ],
            "physical": [
                "🧣 **Warm compress:** Place a warm towel or heating pad on your chest or shoulders.",
                "🎧 **Brown noise:** Put on brown noise or rain sounds. Lower stimulation.",
                "🛋️ **Horizontal:** Lie down flat for 5 minutes. Let your body decompress."
            ],
            "quote": "You don't have to do everything today. Breathe. Prioritize. Let some things wait.",
            "affirmation": "I am allowed to pause. The world will wait for me."
        },
        
        "Scared": {
            "title": "🕯️ Find Safety",
            "color": "warning",
            "quick_relief": [
                "🤝 **Self-holding:** Cross your arms over your chest and give yourself a gentle squeeze (butterfly hug).",
                "🧸 **Safe object:** Hold something soft or comforting. A pillow, a blanket, a stuffed animal.",
                "🏠 **Safe space:** Close your eyes and imagine a place where you feel completely safe. Describe it in detail."
            ],
            "mental_shifts": [
                "🌍 **Check the facts:** 'Am I actually in danger right now, or am I remembering/anticipating danger?'",
                "👤 **Your adult self:** Your younger self needed protection. Your current self can provide it.",
                "📆 **Future self:** Imagine yourself one week from now. This fear will be smaller."
            ],
            "physical": [
                "🦶 **Feet on floor:** Press your feet firmly into the ground. Feel the solid earth beneath you.",
                "🫂 **Weighted blanket:** If available, use a heavy blanket. Deep pressure is calming.",
                "🔦 **Turn on lights:** Darkness amplifies fear. Make your space bright."
            ],
            "quote": "Fear is a compass pointing to what matters. But you don't have to let it drive.",
            "affirmation": "I am safe right now. I can protect myself."
        },
        
        "Worried": {
            "title": "🧘 Quiet the Worry Mind",
            "color": "warning",
            "quick_relief": [
                "📝 **Worry list:** Write down every worry. Then rate each: 'Can I do something about this now?'",
                "⏰ **Worry appointment:** Schedule worry for later. 'I'll worry about this at 6pm for 10 minutes.'",
                "🎲 **Distract:** Do something that requires focus. A puzzle, a game, a recipe."
            ],
            "mental_shifts": [
                "📊 **Probability check:** 'What's the actual likelihood of this happening? What's a more likely outcome?'",
                "🤝 **Friend test:** 'What would I tell a friend who had this worry?'",
                "🔮 **Best/worst/most likely:** What's the best case? Worst case? Most likely? Focus on most likely."
            ],
            "physical": [
                "🫁 **Long exhale:** Breathe in for 4, out for 6. Longer exhales activate the parasympathetic system.",
                "🧴 **Scent grounding:** Use a strong scent - mint, citrus, coffee. Focus entirely on the smell.",
                "👀 **Soft gaze:** Unfocus your eyes and look at the horizon or a blank wall."
            ],
            "quote": "Worrying is like praying for what you don't want to happen.",
            "affirmation": "I release what I cannot control. I handle what I can."
        },
        
        # ----------------------------------------
        # SADNESS & LONELINESS
        # ----------------------------------------
        "Sad": {
            "title": "💙 Gentle Comfort for Sadness",
            "color": "error",
            "quick_relief": [
                "🤗 **Self-hug:** Wrap your arms around yourself. Rock gently side to side.",
                "☕ **Warm comfort:** Make tea, hot chocolate, or soup. Hold the warm cup in both hands.",
                "🎵 **Sad playlist:** Listen to music that matches your mood, then gradually shift to something slightly more upbeat."
            ],
            "mental_shifts": [
                "💬 **Inner critic vs inner friend:** What would you say to a sad friend? Say that to yourself.",
                "📸 **Photo memory:** Look at one photo that brings a small smile. Stay with that feeling for 30 seconds.",
                "🌧️ **Rain metaphor:** Sadness is rain. It waters things. You don't have to like it, but it won't last forever."
            ],
            "physical": [
                "🧣 **Cozy sensory:** Wrap yourself in something soft. A blanket, a hoodie, a scarf.",
                "🕯️ **Candle:** Light a candle and watch the flame. Let your thoughts settle.",
                "🌙 **Rest:** Give yourself permission to do nothing. Sadness is exhausting."
            ],
            "quote": "Sadness is not weakness. It is the price of being human.",
            "affirmation": "I am allowed to feel sad. This feeling will pass."
        },
        
        "Lonely": {
            "title": "🤝 Reconnect",
            "color": "error",
            "quick_relief": [
                "📱 **One message:** Text or call ONE person. Just 'Hey, thinking of you.'",
                "🎧 **Podcast:** Put on a conversation-based podcast. Hear human voices.",
                "🐾 **Pet time:** Spend 5 minutes with an animal. If none, watch animal videos."
            ],
            "mental_shifts": [
                "🌍 **Shared humanity:** Remember: thousands of people feel lonely right now. You are not actually alone in this.",
                "💭 **Quality > Quantity:** One meaningful connection matters more than 100 superficial ones.",
                "☕ **Third place:** Go somewhere with people - a café, library, park. Just be around others."
            ],
            "physical": [
                "🫂 **Weighted pressure:** Use a heavy blanket or stack pillows on your lap.",
                "🌡️ **Warmth:** Take a warm shower or bath. Warmth mimics social connection.",
                "🪑 **Sit outside:** Even 5 minutes in public space can help."
            ],
            "quote": "Loneliness is not the absence of people, but the absence of connection.",
            "affirmation": "I am worthy of connection. I will reach out."
        },
        
        "Empty": {
            "title": "🕳️ Gentle Filling",
            "color": "error",
            "quick_relief": [
                "🎨 **Small creation:** Draw a squiggle, color something, arrange flowers, stack stones.",
                "🎵 **One song:** Find one song that used to make you feel something. Just listen.",
                "🌱 **Tiny task:** Make your bed, water a plant, wash three dishes. Small proof of your impact."
            ],
            "mental_shifts": [
                "🔍 **Curiosity:** 'I feel nothing right now. I wonder what that's about?' No judgment, just observation.",
                "📝 **Sensation check:** What do you notice in your body? Temperature? Tension? Hunger?",
                "⏳ **This is temporary:** Emptiness is not forever. It's a resting state."
            ],
            "physical": [
                "🍊 **Taste something strong:** Lemon, mint, ginger. Focus completely on the flavor.",
                "❄️ **Temperature change:** Step outside. Cold air can help you feel present.",
                "💪 **Gentle movement:** Stretch, shake your hands, roll your neck."
            ],
            "quote": "Emptiness is not a void to fill, but space to notice.",
            "affirmation": "I am here. That is enough."
        },
        
        "Tired": {
            "title": "😴 True Rest",
            "color": "info",
            "quick_relief": [
                "⏸️ **Guilt-free pause:** Rest is productive. Set a timer for 10 minutes. Do absolutely nothing.",
                "💤 **Power nap:** 20 minutes max. Set an alarm. Even just lying down helps.",
                "🫁 **Yawn:** Fake yawn until a real one comes. It signals your nervous system to relax."
            ],
            "mental_shifts": [
                "🚫 **Permission slip:** 'I give myself permission to not be productive right now.'",
                "📉 **Lower the bar:** Today's goal is survival and rest. That's enough.",
                "💭 **Not lazy, just tired:** You're not failing. You're depleted. These are different things."
            ],
            "physical": [
                "💧 **Hydrate:** Fatigue is often dehydration. Drink a full glass of water.",
                "🦵 **Legs up the wall:** Lie on floor, rest legs against wall. 5 minutes. Restores blood flow.",
                "🌿 **Dim lights:** Lower screen brightness, turn off overhead lights."
            ],
            "quote": "You are not a machine. Rest is not a reward for exhaustion—it's a necessity.",
            "affirmation": "I am allowed to rest. My worth is not my productivity."
        },
        
        # ----------------------------------------
        # ANGER & FRUSTRATION
        # ----------------------------------------
        "Angry": {
            "title": "🔥 Release Anger Safely",
            "color": "error",
            "quick_relief": [
                "🏃 **Move your body:** 1 minute of jumping jacks, running in place, or shaking your whole body.",
                "✍️ **Write & destroy:** Type everything you're angry about, then delete it. Or write on paper and tear it up.",
                "🧊 **Cold shock:** Hold an ice cube. Splash cold water on your face. The cold resets your nervous system."
            ],
            "mental_shifts": [
                "🗣️ **Name it:** 'I am angry because ______.' Speaking the reason reduces its power.",
                "🔍 **Beneath the anger:** Anger is often a protector. What's underneath? Hurt? Fear? Injustice?",
                "🚶 **Walk away:** It's okay to remove yourself. You can return to the conversation later."
            ],
            "physical": [
                "🦷 **Check your jaw:** Are you clenching? Unclench. Let your tongue rest at the bottom of your mouth.",
                "👊 **Progressive relaxation:** Squeeze fists tight → hold → release. Feel the release.",
                "🧴 **Cold wrists:** Run cold water over your wrists for 30 seconds."
            ],
            "quote": "Anger is a messenger. Listen to what it's telling you, then let it go.",
            "affirmation": "I am in control of how I respond. I choose calm."
        },
        
        "Frustrated": {
            "title": "😤 Unblock & Reset",
            "color": "error",
            "quick_relief": [
                "⏸️ **Step away:** Physically leave the frustrating situation. 5 minutes minimum.",
                "🔄 **Switch tasks:** Do something completely different for 10 minutes, then return.",
                "🎯 **Small win:** Find ONE tiny thing you can complete successfully. Check it off."
            ],
            "mental_shifts": [
                "🌊 **'This too shall pass'** - Frustration is temporary. It will peak and subside.",
                "🤔 **What's the block?** Is it skill? Resource? Support? Identify it specifically.",
                "📉 **Lower expectations:** You're allowed to struggle. Struggle is learning."
            ],
            "physical": [
                "😤 **Sigh:** Exaggerated, loud sigh. Do it 3 times. It releases tension.",
                "🙆 **Open chest:** Clasp hands behind back, open chest, breathe deeply.",
                "🦋 **Butterfly taps:** Cross arms, tap alternately on chest. Bilateral stimulation calms."
            ],
            "quote": "Frustration means you care. Use it as fuel, not friction.",
            "affirmation": "I can handle setbacks. I will try again."
        },
        
        # ----------------------------------------
        # GUILT & SHAME
        # ----------------------------------------
        "Guilty": {
            "title": "🕊️ Release Guilt",
            "color": "error",
            "quick_relief": [
                "📝 **Name the mistake:** Write what happened. 'I did X. It had Y impact. I feel guilty.'",
                "🤝 **Make amends (small):** If possible, one small repair. A message, an apology, a fix.",
                "💬 **Talk to yourself like a friend:** 'You made a mistake. You're still a good person.'"
            ],
            "mental_shifts": [
                "📊 **Is this guilt or shame?** Guilt = 'I did something bad.' Shame = 'I am bad.' You did, you are not.",
                "🌱 **Learn vs. Ruminate:** What can you learn? What will you do differently? Done.",
                "⏪ **You did what you could with what you knew.** Hindsight is clearer. Be kind to your past self."
            ],
            "physical": [
                "🫂 **Self-compassion break:** Hand on heart. 'This is hard. I'm not alone. May I be kind to myself.'",
                "🧘 **Open posture:** Don't curl inward. Open your chest, sit upright.",
                "💧 **Wash hands:** Symbolically wash away the mistake."
            ],
            "quote": "Guilt says you've done something wrong. Shame says you ARE wrong. One is true. One is not.",
            "affirmation": "I am not my mistakes. I can grow and do better."
        },
        
        "Ashamed": {
            "title": "🕊️ Self-Forgiveness",
            "color": "error",
            "quick_relief": [
                "🫂 **Butterfly hug:** Cross arms, tap alternately on chest. 20 taps. 'I am worthy of compassion.'",
                "📝 **Write a forgiveness letter:** To yourself. You don't have to send it. Just write it.",
                "🧸 **Inner child:** Imagine your younger self. What do they need to hear? Tell them."
            ],
            "mental_shifts": [
                "🔓 **Shame thrives in secrecy.** Share with one safe person. 'I feel ashamed about...'",
                "🌍 **You are not alone.** Almost everyone carries shame. It's a human experience, not a personal failing.",
                "🌟 **Separate action from identity:** You did something vs. You are something. Choose the first."
            ],
            "physical": [
                "🫀 **Heart focus:** Place both hands on your heart. Feel the warmth. Breathe into your heart center.",
                "🧣 **Wrap yourself:** Physical warmth and pressure. A blanket, a coat, a hug.",
                "👁️ **Soft gaze:** Look in a mirror and say your name. 'I see you. You are enough.'"
            ],
            "quote": "Shame cannot survive empathy. Be the first person to offer yourself some.",
            "affirmation": "I am worthy of love and belonging, exactly as I am."
        },
        
        # ----------------------------------------
        # POSITIVE EMOTIONS
        # ----------------------------------------
        "Happy": {
            "title": "🌿 Savor This Moment",
            "color": "success",
            "quick_relief": [
                "📸 **Capture it:** Write down exactly what's making you happy right now. Be specific.",
                "🧠 **30-second savor:** Close your eyes and really feel this happiness for 30 seconds. Where is it in your body?",
                "🙏 **Gratitude ping:** Text one person something you appreciate about them. Right now."
            ],
            "mental_shifts": [
                "💭 **'This is happiness.'** Label the moment. Not 'I was happy' or 'I will be happy' - I AM happy.",
                "📝 **Joy log:** Add this to your mental list of things that bring you joy. Come back to it later.",
                "🎁 **Happiness is not a destination.** It's moments like this. You found one."
            ],
            "physical": [
                "😊 **Smile:** Even if forced, smiling signals safety to your brain. Hold it for 20 seconds.",
                "🌞 **Look up:** Tilt your chin up, open your posture. Let yourself receive this moment.",
                "🫁 **Breathe it in:** Deep breath, imagine breathing in this good feeling."
            ],
            "quote": "Happiness is not something you postpone for the future; it is something you design for the present.",
            "affirmation": "I deserve to feel good. I welcome joy."
        },
        
        "Calm": {
            "title": "🧘 Deepen Your Calm",
            "color": "success",
            "quick_relief": [
                "🌊 **Ride the wave:** Notice this calm feeling. Don't cling to it, just observe it.",
                "☕ **Slow moment:** Do one thing slowly. Drink water slowly. Walk slowly. Breathe slowly.",
                "👂 **Listen:** Close your eyes. What's the quietest sound you can hear?"
            ],
            "mental_shifts": [
                "💭 **'This is enough.'** You don't need to chase excitement. Calm is complete.",
                "🧘 **Stillness practice:** 2 minutes of doing absolutely nothing. You've already succeeded.",
                "📚 **Calm is a resource:** Store this feeling. Remember it. You can return here."
            ],
            "physical": [
                "🫁 **Even breath:** Inhale 4, exhale 4. No pause needed. Just even, steady breath.",
                "👐 **Open palms:** Turn palms up on your lap. Receptive, open, calm.",
                "🧣 **Soft eyes:** Relax your gaze. Soften your face."
            ],
            "quote": "Peace is not the absence of chaos, but the presence of calm in the midst of it.",
            "affirmation": "I am at peace with this moment."
        },
        
        "Grateful": {
            "title": "🙏 Amplify Gratitude",
            "color": "success",
            "quick_relief": [
                "📝 **Three good things:** Write down three things that went well today. Any size.",
                "💌 **Gratitude note:** Text or write a short note to someone. 'I'm grateful for you because...'",
                "🧠 **Notice:** What are you grateful for RIGHT NOW? This device? Warmth? Breath?"
            ],
            "mental_shifts": [
                "🔄 **'I get to' not 'I have to'** - Reframe one obligation as a privilege.",
                "🔍 **Find the small:** Not just big things. The way light hits the wall. The first sip. Comfortable shoes.",
                "🌱 **Gratitude grows:** The more you look for it, the more you find."
            ],
            "physical": [
                "🫀 **Heart focus:** Hand on heart. Breathe gratitude into your chest.",
                "😊 **Grateful smile:** Smile as you think of one thing you're thankful for.",
                "🌞 **Look up:** Literally look up. It shifts your nervous system toward openness."
            ],
            "quote": "Gratitude turns what we have into enough.",
            "affirmation": "I have so much to be grateful for. I choose to notice it."
        },
        
        "Hopeful": {
            "title": "✨ Nurture Hope",
            "color": "success",
            "quick_relief": [
                "🔮 **Future visualization:** Imagine one good thing, even small, that might happen tomorrow.",
                "📝 **Possibility list:** Write down 3 things you're looking forward to. Any size.",
                "🌅 **Sunrise/sunset:** Look at the sky. Hope is natural. The sun always rises."
            ],
            "mental_shifts": [
                "💭 **Hope is a practice.** Not naive optimism. Just the belief that good things are possible.",
                "🛤️ **Small steps:** What's one tiny step toward something you hope for?",
                "🤝 **Share it:** Tell someone about something you're hopeful about. Hope grows when shared."
            ],
            "physical": [
                "🌞 **Light:** Get sunlight on your skin, especially morning light. Signals hope biologically.",
                "🫁 **Open breath:** Breathe into your upper chest. Open, expansive posture.",
                "👣 **Forward movement:** Walk forward, even a few steps. Literally moving forward."
            ],
            "quote": "Hope is not the conviction that something will turn out well, but the certainty that something makes sense regardless of how it turns out.",
            "affirmation": "I believe good things are possible. I am open to hope."
        },
        
        # ----------------------------------------
        # NEUTRAL
        # ----------------------------------------
        "Okay": {
            "title": "🌱 Gentle Check-In",
            "color": "info",
            "quick_relief": [
                "💭 **'Okay is okay.'** You don't need to feel great to have a good day.",
                "📝 **Small win:** What's one thing that went okay today? Acknowledge it.",
                "🧘 **Presence:** You're here. You're reading this. That's enough."
            ],
            "mental_shifts": [
                "📊 **'Okay' is stable.** Not every day needs to be amazing. Stability is valuable.",
                "🎯 **One thing:** What's one small thing you want to do today? Just one.",
                "🌿 **Contentment practice:** Can you be content with 'okay' right now?"
            ],
            "physical": [
                "🫁 **Neutral breath:** No special pattern. Just notice your breath without changing it.",
                "🧣 **Comfort check:** Adjust something for comfort. Temperature? Position? Clothing?",
                "👁️ **Soft gaze:** Look at something neutral. A wall. The sky. Rest your eyes."
            ],
            "quote": "You don't have to feel amazing every day. 'Okay' is enough.",
            "affirmation": "I am okay. That is enough for today."
        },
        
        "Neutral": {
            "title": "🌱 Gentle Presence",
            "color": "info",
            "quick_relief": [
                "🧘 **Just notice:** Don't try to change your mood. Just notice it without judgment.",
                "📝 **Check-in:** 'I feel neutral. That's fine. It will shift naturally.'",
                "🎯 **One small action:** What's one tiny thing you want to do? Do it."
            ],
            "mental_shifts": [
                "📊 **Neutral is not bad.** It's not negative. It's just... neutral. That's allowed.",
                "🌊 **Moods shift.** This won't last forever. Neither will the hard days. Neither will the good ones.",
                "💭 **Curiosity:** 'I wonder what I'll feel later?' No pressure, just curiosity."
            ],
            "physical": [
                "🫁 **Three breaths:** Three intentional breaths. No count, no pattern. Just breath.",
                "🧣 **Comfort:** Adjust something small. Socks? Lighting? Posture?",
                "👣 **Ground:** Feel your feet on the floor. You are here."
            ],
            "quote": "Still water runs deep. You don't have to be in motion to have depth.",
            "affirmation": "I am present. That is enough."
        }
    }
    
    # Return suggestions for the emotion, or a default
    return suggestions.get(emotion, {
        "title": "🌱 Check-In",
        "color": "info",
        "quick_relief": [
            "💭 **What do I need right now?** Rest? Connection? Movement? Silence?",
            "🎯 **What's one tiny step I can take?**",
            "🤔 **If I felt better, what would be different?**"
        ],
        "mental_shifts": [
            "📝 **Name the emotion:** What am I actually feeling?",
            "🌊 **This too shall pass.** All feelings are temporary.",
            "💬 **Talk to yourself like a friend.** What would you say?"
        ],
        "physical": [
            "🫁 **Three deep breaths.** In through nose, out through mouth.",
            "💧 **Drink water.** Your brain needs it.",
            "🧣 **Adjust for comfort.** Temperature, position, lighting."
        ],
        "quote": "Small steps are still progress.",
        "affirmation": "I am doing the best I can. That is enough."
    })

# ---------- SIDEBAR - BLACK MAC-FRIENDLY NAVIGATION ----------
st.sidebar.markdown("""
<style>
    /* Clean, black styling for Mac */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        padding: 20px 0px;
    }
    
    /* App title - white on black */
    .sidebar-title {
        font-size: 28px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
        padding-left: 20px;
    }
    
    /* BIG DATE STYLING - black card */
    .big-date {
        background: #1a1a1a;
        padding: 20px 20px;
        border-radius: 16px;
        margin: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #333333;
    }
    .day-name {
        font-size: 18px;
        font-weight: 500;
        color: #b0b0b0;
        margin-bottom: 4px;
        letter-spacing: -0.2px;
    }
    .day-date {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    .month-year {
        font-size: 18px;
        font-weight: 500;
        color: #b0b0b0;
        margin-top: 4px;
    }
    
    /* Clean nav buttons - black theme */
    .nav-item {
        padding: 10px 20px;
        margin: 2px 0px;
        border-radius: 0px;
        font-weight: 500;
        color: #e0e0e0;
        transition: all 0.1s;
        display: flex;
        align-items: center;
        gap: 12px;
        border-left: 3px solid transparent;
    }
    .nav-item:hover {
        background-color: #2a2a2a;
        border-left: 3px solid #808080;
        color: #ffffff;
    }
    .nav-item-active {
        background-color: #1a33a8 !important;
        border-left: 3px solid #4d94ff !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    .nav-icon {
        font-size: 20px;
        width: 28px;
        text-align: center;
    }
    
    /* Stats card - black theme */
    .stats-card {
        background: #1a1a1a;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 8px 20px;
        border: 1px solid #333333;
    }
    .stats-label {
        font-size: 14px;
        color: #b0b0b0;
        margin-bottom: 4px;
    }
    .stats-number {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Divider - dark grey */
    .sidebar-divider {
        margin: 16px 20px;
        border-top: 1px solid #333333;
    }
    
    /* Expander in sidebar - black theme */
    .streamlit-expanderHeader {
        color: #e0e0e0 !important;
        background-color: #1a1a1a !important;
    }
    .streamlit-expanderContent {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    /* Make primary button greener */
    .stButton button[kind="primary"] {
        background-color: #0B6623 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #0A5C1F !important;
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# App title
st.sidebar.markdown('<div class="sidebar-title">🌱 MindEase</div>', unsafe_allow_html=True)

# ---------- BIG DATE DISPLAY ----------
today = date.today()
st.sidebar.markdown(f'''
<div class="big-date">
    <div class="day-name">{today.strftime("%A")}</div>
    <div class="day-date">{today.strftime("%d")}</div>
    <div class="month-year">{today.strftime("%B %Y")}</div>
</div>
''', unsafe_allow_html=True)

# Navigation options - ONLY 4 PAGES
nav_items = [
    {"icon": "📝", "label": "Journal", "id": "journal"},
    {"icon": "📊", "label": "History", "id": "history"},
    {"icon": "🧰", "label": "Self-Care", "id": "selfcare"},
    {"icon": "🆘", "label": "Crisis Help", "id": "crisis"}
]

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "journal"

# Create navigation buttons
for item in nav_items:
    # Create columns for icon and label
    cols = st.sidebar.columns([1, 5])
    
    with cols[0]:
        st.markdown(f'<div class="nav-icon">{item["icon"]}</div>', unsafe_allow_html=True)
    
    with cols[1]:
        if st.button(
            item["label"],
            key=f"nav_{item['id']}",
            use_container_width=True
        ):
            st.session_state.current_page = item["id"]
            st.rerun()

# Quick stats
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

df = load_log()
if not df.empty:
    today_entries = df[df["date"] == today.isoformat()]
    
    st.sidebar.markdown('<div class="stats-card">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="stats-label">Today\'s entries</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="stats-number">{len(today_entries)}</div>', unsafe_allow_html=True)
    
    if not today_entries.empty:
        latest_mood = today_entries.iloc[-1]["mood"]
        emoji = EMOTION_DATA.get(latest_mood, {}).get("emoji", "📝")
        st.sidebar.markdown(f'<div style="margin-top: 12px; font-size: 16px; color: white;">Latest: {emoji} {latest_mood}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="stats-card">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="stats-label">Welcome!</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="font-size: 16px; margin-top: 8px; color: white;">Start your first entry ✨</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Privacy footer
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
with st.sidebar.expander("🔒 Private & Local", expanded=False):
    st.markdown("""
    • No login needed  
    • Data stays on your device  
    • Not medical advice  
    • 24/7 crisis support available
    """)

# ---------- SET CURRENT PAGE BASED ON NAVIGATION ----------
page_map = {
    "journal": "📝 Journal",
    "history": "📊 History",
    "selfcare": "🧰 Self-Care",
    "crisis": "🆘 Crisis Help"
}
page = page_map[st.session_state.current_page]

# ---------- MAIN CONTENT ----------
# ---------- PAGE 1: JOURNAL - UNIFIED, AUTO-ANALYZE (NO CHATBOT) ----------
if page == "📝 Journal":
    st.title("📝 MindEase Journal")
    st.caption(f"Today • {today.strftime('%A, %d %B %Y')}")
    
    # Single journal text area
    journal_entry = st.text_area(
        "How are you feeling?",
        placeholder="Write anything... I'll understand how you're feeling and suggest ways to help.",
        height=200
    )
    
    # Character counter
    if journal_entry:
        words = len(journal_entry.split())
        chars = len(journal_entry)
        st.caption(f"📝 {words} words • {chars} characters")
    
    # Green save & analyze button
    save_button = st.button("💾 Save & Analyze Entry", use_container_width=True, type="primary", key="save_analyze")
    
    if save_button:
        if not journal_entry.strip():
            st.warning("Please write something before saving.")
        else:
            # Check for crisis keywords FIRST
            if check_crisis_keywords(journal_entry):
                display_crisis_support()
                # Save as crisis entry
                save_entry("Crisis", journal_entry, source="crisis")
                st.success("✅ Entry saved. Please reach out for support.")
            else:
                # Analyze with Gemini
                with st.spinner("🧠 Analyzing your emotions..."):
                    emotion, explanation = analyze_with_gemini(journal_entry)
                
                # Display result
                st.divider()
                display_emotion_result(emotion, explanation)
                
                # Save to history
                save_entry(emotion, journal_entry, source="ai")
                st.success("✅ Journal saved and analyzed!")
                
                # ---------- SELF-CARE SUGGESTIONS ----------
                st.divider()
                
                # Get personalized suggestions
                care = get_self_care_suggestions(emotion)
                
                # Display header with emotion-specific color
                if care["color"] == "success":
                    st.success(f"💚 **{care['title']}**")
                elif care["color"] == "warning":
                    st.warning(f"💛 **{care['title']}**")
                elif care["color"] == "error":
                    st.error(f"❤️ **{care['title']}**")
                else:
                    st.info(f"💙 **{care['title']}**")
                
                # Create three columns for different types of help
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### ⚡ Quick Relief")
                    for item in care["quick_relief"][:3]:
                        st.markdown(item)
                
                with col2:
                    st.markdown("#### 🧠 Mind Shift")
                    for item in care["mental_shifts"][:2]:
                        st.markdown(item)
                
                with col3:
                    st.markdown("#### 🫂 Body")
                    for item in care["physical"][:2]:
                        st.markdown(item)
                
                # Quote and affirmation
                st.divider()
                st.markdown(f"> *{care['quote']}*")
                st.markdown(f"**✨ Today's affirmation:** {care['affirmation']}")
                
                # See all suggestions expander
                with st.expander("📚 See all suggestions for this emotion"):
                    st.markdown("#### ⚡ Quick Relief")
                    for item in care["quick_relief"]:
                        st.markdown(item)
                    
                    st.markdown("#### 🧠 Mind Shift")
                    for item in care["mental_shifts"]:
                        st.markdown(item)
                    
                    st.markdown("#### 🫂 Body")
                    for item in care["physical"]:
                        st.markdown(item)
    
    # Show recent entries
    st.divider()
    with st.expander("📋 Your Recent Journal Entries", expanded=False):
        df = load_log()
        if not df.empty:
            recent_entries = df.sort_values(by="date", ascending=False).head(5)
            for _, row in recent_entries.iterrows():
                mood = row["mood"]
                emoji = EMOTION_DATA.get(mood, {}).get("emoji", "📝")
                
                # Source icon
                if row["source"] == "crisis":
                    source_icon = "🚨"
                else:
                    source_icon = "🤖"
                
                st.markdown(f"**{emoji} {mood}** {source_icon} • {row['date']}")
                st.caption(row["note"][:100] + "..." if len(row["note"]) > 100 else row["note"])
                st.divider()
        else:
            st.info("No journal entries yet. Write your first entry above!")

# ---------- PAGE 2: HISTORY ----------
elif page == "📊 History":
    st.title("📊 Your Mood History")
    
    df = load_log()
    if df.empty:
        st.info("No entries yet. Start by journaling!")
    else:
        # Convert date column
        df["date"] = pd.to_datetime(df["date"])
        
        # Filter by date range
        col1, col2 = st.columns(2)
        with col1:
            filter_option = st.selectbox(
                "Show",
                ["All time", "Last 7 days", "Last 30 days", "This month"],
                index=0
            )
        
        # Apply date filter
        filtered_df = df.copy()
        if filter_option == "Last 7 days":
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
            filtered_df = filtered_df[filtered_df["date"] >= cutoff]
        elif filter_option == "Last 30 days":
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            filtered_df = filtered_df[filtered_df["date"] >= cutoff]
        elif filter_option == "This month":
            filtered_df = filtered_df[filtered_df["date"].dt.month == pd.Timestamp.now().month]
        
        with col2:
            source_filter = st.selectbox(
                "Type",
                ["All entries", "Journal only"],
                index=0
            )
        
        if source_filter == "Journal only":
            filtered_df = filtered_df[filtered_df["source"] == "ai"]
        
        # Sort by date
        filtered_df = filtered_df.sort_values("date", ascending=False)
        
        # Metrics
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total entries", len(filtered_df))
        with col2:
            if not filtered_df.empty:
                avg_score = filtered_df["score"].mean()
                st.metric("Average mood", f"{avg_score:.1f}/5")
        with col3:
            if not filtered_df.empty:
                most_common = filtered_df["mood"].mode()[0] if not filtered_df["mood"].mode().empty else "N/A"
                emoji = EMOTION_DATA.get(most_common, {}).get("emoji", "")
                st.metric("Most frequent", f"{emoji} {most_common}")
        with col4:
            if not filtered_df.empty:
                today_count = len(filtered_df[filtered_df["date"].dt.date == date.today()])
                st.metric("Today", today_count)
        
        # Mood trend chart
        st.subheader("Mood Trend")
        chart_df = filtered_df.sort_values("date")
        if not chart_df.empty:
            st.line_chart(chart_df.set_index("date")["score"])
        
        # Emotion distribution
        st.subheader("Emotion Distribution")
        mood_counts = filtered_df["mood"].value_counts().head(8)
        
        # Add emojis to labels
        mood_labels = []
        for mood in mood_counts.index:
            emoji = EMOTION_DATA.get(mood, {}).get("emoji", "📌")
            mood_labels.append(f"{emoji} {mood}")
        
        # Create bar chart
        chart_data = pd.DataFrame({
            "Emotion": mood_labels,
            "Count": mood_counts.values
        })
        st.bar_chart(chart_data.set_index("Emotion"))
        
        # Detailed entries
        st.subheader("Recent Entries")
        for _, row in filtered_df.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    mood = row["mood"]
                    emoji = EMOTION_DATA.get(mood, {}).get("emoji", "📝")
                    
                    if row["source"] == "crisis":
                        source_icon = "🚨"
                    else:
                        source_icon = "🤖"
                    
                    st.markdown(f"**{emoji}**")
                    st.caption(f"{source_icon}")
                with col2:
                    st.markdown(f"**{mood}** • {row['date'].strftime('%b %d, %Y')}")
                    if pd.notna(row["note"]) and row["note"]:
                        st.caption(row["note"][:150] + "..." if len(row["note"]) > 150 else row["note"])
                st.divider()
        
        # Clear data button
        with st.expander("⚠️ Data Management"):
            st.warning("This will permanently delete all your saved journal entries.")
            if st.button("Clear All Data", type="secondary"):
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                st.success("All entries cleared.")
                st.rerun()

# ---------- PAGE 3: SELF-CARE ----------
elif page == "🧰 Self-Care":
    st.title("🧰 Self-Care Toolbox")
    st.caption("Quick exercises to help you feel grounded")
    
    tabs = st.tabs(["🌬️ Breathing", "👁️ Grounding", "💬 Affirmations", "📝 Prompts"])
    
    with tabs[0]:  # Breathing
        st.subheader("Box Breathing")
        st.write("**4-4-4-4 Technique:** Inhale → Hold → Exhale → Hold")
        
        cycles = st.slider("Number of cycles", 2, 8, 4)
        
        if st.button("Start Breathing Exercise", use_container_width=True):
            with st.spinner(f"Breathing exercise in progress..."):
                import time
                progress_bar = st.progress(0)
                total_steps = cycles * 4
                for i in range(total_steps):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / total_steps)
            st.success("✨ Complete! Notice how you feel now.")
    
    with tabs[1]:  # Grounding
        st.subheader("5-4-3-2-1 Grounding")
        st.write("Name...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("5 things you can **SEE**", placeholder="lamp, window, cup...")
            st.text_input("4 things you can **FEEL**", placeholder="chair, fabric, breeze...")
            st.text_input("3 things you can **HEAR**", placeholder="fan, birds, typing...")
        with col2:
            st.text_input("2 things you can **SMELL**", placeholder="coffee, fresh air...")
            st.text_input("1 thing you can **TASTE**", placeholder="mint, water...")
        
        st.caption("This exercise brings you back to the present moment.")
    
    with tabs[2]:  # Affirmations
        st.subheader("Daily Affirmations")
        
        affirmations = [
            "I am doing the best I can, and that is enough.",
            "My feelings are valid, and they will pass.",
            "I deserve kindness, especially from myself.",
            "It's okay to ask for help when I need it.",
            "I don't have to be perfect to be worthy.",
            "This moment is temporary. I can get through this.",
            "I am allowed to rest. I am allowed to heal."
        ]
        
        if st.button("✨ Show me an affirmation", use_container_width=True):
            import random
            affirmation = random.choice(affirmations)
            st.success(f"💛 {affirmation}")
    
    with tabs[3]:  # Prompts
        st.subheader("Gentle Journal Prompts")
        
        prompts = [
            "What's one thing I can control today?",
            "What small moment brought me peace recently?",
            "If my best friend felt this way, what would I tell them?",
            "What's one thing I did well today, even if small?",
            "What do I need right now, in this moment?",
            "What's something I'm looking forward to?"
        ]
        
        for prompt in prompts:
            st.markdown(f"- {prompt}")

# ---------- PAGE 4: CRISIS HELP ----------
else:
    st.title("🆘 Crisis Support - Singapore")
    st.caption("24/7 confidential helplines")
    
    display_safety_block()
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧑‍🤝‍🧑 For Youth
        - **CHAT (youth mental health)**: 6493 6500
        - **Tinkle Friend**: 1800 274 4788
        - **eCounselling Centre**: chat.mentalhealth.sg
        
        ### 👨‍👩‍👧 For Families
        - **Care Corner Counselling**: 1800 353 5800
        - **Fei Yue Family Service**: 6819 9170
        """)
    
    with col2:
        st.markdown("""
        ### 🏥 Professional Help
        - **IMH Appointment**: 6389 2000
        - **SGH Psychiatry**: 6321 4377
        - **NUH Psychiatry**: 6772 2000
        
        ### 💻 Online Support
        - **SOS Chat**: sos.org.sg
        - **Mindline.sg**: mindline.sg
        - **EC2 (SAMH)**: samhealth.org.sg
        """)
    
    st.divider()
    st.warning("""
    **If you or someone else is in immediate danger, please call 995 now.**
    
    These helplines are confidential and staffed by trained professionals.
    You don't need to be in crisis to call - they're here to listen.
    """)
