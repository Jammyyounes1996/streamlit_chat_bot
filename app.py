import streamlit as st
import google.generativeai as genai
from transformers import pipeline
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Smart Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_used" not in st.session_state:
    st.session_state.model_used = None

if "huggingface_model" not in st.session_state:
    st.session_state.huggingface_model = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    gemini_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        help="Get your key from https://makersuite.google.com/app/apikey"
    )
    
    model_choice = st.radio(
        "Select Model",
        ["Auto (Gemini → HuggingFace)", "Force Gemini", "Force HuggingFace"]
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.model_used = None
        st.rerun()
    
    st.divider()
    st.write("### Status")
    if st.session_state.model_used:
        st.success(f"Using: {st.session_state.model_used}")
    else:
        st.info("Ready to chat...")

# Initialize Gemini
def init_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                'temperature': temperature,
                'top_p': 0.95,
                'top_k': 64,
            }
        )
        return model
    except Exception as e:
        st.error(f"❌ Gemini initialization failed: {str(e)}")
        return None

# Initialize HuggingFace fallback
@st.cache_resource
def init_huggingface():
    try:
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1  # CPU
        )
        return pipe
    except Exception as e:
        st.error(f"❌ HuggingFace initialization failed: {str(e)}")
        return None

# Get response from Gemini
def get_gemini_response(user_input, conversation_history, model):
    try:
        messages = []
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "parts": [msg["content"]]
            })
        
        messages.append({"role": "user", "parts": [user_input]})
        
        response = model.generate_content(
            messages,
            stream=False
        )
        
        return response.text if response.text else "Sorry, I couldn't generate a response."
    except Exception as e:
        return None

# Get response from HuggingFace
def get_huggingface_response(user_input, pipe):
    try:
        prompt = f"User: {user_input}\nAssistant:"
        
        response = pipe(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            truncation=True
        )
        
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.split("Assistant:")[-1].strip()
        
        return assistant_response if assistant_response else "I'm not sure how to respond to that."
    except Exception as e:
        st.error(f"Error with HuggingFace: {str(e)}")
        return None

# Main chat function
def chat(user_input):
    # Determine which model to use
    use_gemini = False
    use_huggingface = False
    
    if model_choice == "Force Gemini":
        use_gemini = True
    elif model_choice == "Force HuggingFace":
        use_huggingface = True
    else:  # Auto mode
        use_gemini = bool(gemini_key)
        use_huggingface = True  # Always have HF as fallback
    
    response = None
    model_name = None
    
    # Try Gemini first
    if use_gemini and gemini_key:
        gemini_model = init_gemini(gemini_key)
        if gemini_model:
            response = get_gemini_response(
                user_input,
                st.session_state.messages,
                gemini_model
            )
            if response:
                model_name = "🟦 Gemini 1.5 Flash"
    
    # Fallback to HuggingFace
    if not response and use_huggingface:
        if st.session_state.huggingface_model is None:
            with st.spinner("Loading HuggingFace model..."):
                st.session_state.huggingface_model = init_huggingface()
        
        if st.session_state.huggingface_model:
            response = get_huggingface_response(
                user_input,
                st.session_state.huggingface_model
            )
            if response:
                model_name = "🤗 DistilGPT-2 (Fallback)"
    
    if not response:
        response = "Sorry, I couldn't process your request. Please check your API key or try again."
        model_name = "❌ Error"
    
    return response, model_name

# Main UI
st.title("🤖 Smart Chatbot")
st.markdown("**Multi-Model Chat Assistant** - Gemini API with HuggingFace Fallback")

st.divider()

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            model_badge = message.get("model", "Assistant")
            st.markdown(
                f'<div class="chat-message assistant-message"><strong>Assistant ({model_badge}):</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )

st.divider()

# Input section
col1, col2 = st.columns([0.85, 0.15])

with col1:
    user_input = st.text_input(
        "Your message:",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True)

# Process input
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Get response
    with st.spinner("🤔 Thinking..."):
        response, model_used = chat(user_input)
        st.session_state.model_used = model_used
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "model": model_used
    })
    
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 2rem;'>
    <small>Made with ❤️ using Streamlit | Powered by Gemini API & HuggingFace</small>
</div>
""", unsafe_allow_html=True)