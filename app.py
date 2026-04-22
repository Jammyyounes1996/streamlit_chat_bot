import streamlit as st
import google.generativeai as genai
import requests
import json

# Page config
st.set_page_config(page_title="AI Chat", layout="wide")
st.title("🤖 AI Chatbot")

# Sidebar for API key
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key:", type="password", key="api_key")
    hf_token = st.text_input("HuggingFace Token (Optional):", type="password", key="hf_token")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = None

# Configure Gemini
def load_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        st.session_state.model_name = "Gemini (✓)"
        return True
    except Exception as e:
        st.warning(f"Gemini config failed: {str(e)}")
        return False

# Query HuggingFace API
def query_huggingface(prompt, hf_token):
    try:
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        payload = {"inputs": prompt, "parameters": {"max_length": 200}}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"HF API Error: {response.status_code}"
    except Exception as e:
        return f"HuggingFace error: {str(e)}"

# Get response
def get_response(user_input, api_key, hf_token):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_input)
        st.session_state.model_name = "Gemini (✓)"
        return response.text
    except Exception as e:
        st.warning(f"Gemini failed. Trying HuggingFace...")
        st.session_state.model_name = "HuggingFace (Fallback)"
        return query_huggingface(user_input, hf_token)

# Load Gemini on startup
if api_key and not st.session_state.model_name:
    with st.spinner("Loading Gemini..."):
        load_gemini(api_key)

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start chatting")
else:
    # Model status
    if st.session_state.model_name:
        st.success(f"✅ Using: {st.session_state.model_name}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_input, api_key, hf_token)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()