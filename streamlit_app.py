import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")
    return tokenizer, model

tokenizer, model = load_model()


labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
          'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
          'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
          'remorse', 'sadness', 'surprise', 'neutral']

PERSONAS = {
    "joy": "😄 I'm so glad to hear that! What's bringing you joy today?",
    "sadness": "💙 I'm here for you. Want to talk about what's making you feel this way?",
    "anger": "😤 That sounds frustrating. You have every right to feel upset.",
    "love": "❤️ Love is powerful. Who or what made you feel this way?",
    "fear": "😨 That must be scary. Do you want to share more?",
    "caring": "🤗 That's very kind of you. Your empathy is beautiful.",
    "amusement": "😂 You're hilarious! Got more jokes?",
    "gratitude": "🙏 You're welcome! It's always a pleasure to help.",
    "neutral": "🙂 I'm listening. Tell me more.",
    "surprise": "😲 Whoa! That must have been unexpected!"
}


def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    top_index = torch.argmax(probs, dim=1).item()
    emotion = labels[top_index]
    return emotion


def generate_response(text):
    emotion = detect_emotion(text)
    response = PERSONAS.get(emotion, "🙂 I'm here. Tell me more.")
    return emotion, response


st.set_page_config(page_title="Emotion-Aware Chatbot 🤖", page_icon="🧠")
st.title("🧠 Emotion-Aware Chatbot")
st.markdown("Chat with a bot that understands your **emotions** and responds accordingly!")


user_input = st.text_input("You:", placeholder="How are you feeling today?")


if user_input:
    emotion, response = generate_response(user_input)
    st.markdown(f"**Detected Emotion:** `{emotion}`")
    st.markdown(f"**Bot:** {response}")

