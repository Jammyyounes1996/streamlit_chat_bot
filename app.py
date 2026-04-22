import streamlit as st
import google.generativeai as genai
import requests

# 1. إعدادات الصفحة بتصميم نظيف
st.set_page_config(page_title="AI Chatbot", layout="centered")

# تحسين الخط وشكل الواجهة (Apple-style simplicity)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Chatbot")

# 2. الإعدادات في الشريط الجانبي
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key:", type="password", key="api_key_input")
    hf_token = st.text_input("HuggingFace Token (Optional):", type="password", key="hf_token_input")
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. تهيئة الجلسة (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. دالة الاستعلام من HuggingFace (رابط محدث)
def query_huggingface(prompt, hf_token):
    try:
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        # استخدام موديل Mistral v0.3 أو Zephyr لأنهما أكثر استقراراً في الـ Free API
        API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            # معالجة استجابة HF لأنها ترجع قائمة
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            return str(result)
        else:
            return f"HuggingFace Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"HF Connection Failed: {str(e)}"

# 5. دالة الحصول على الرد (المنطق الرئيسي)
def get_ai_response(user_input, api_key, hf_token):
    # محاولة تشغيل Gemini أولاً
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
            response = model.generate_content(user_input)
            return response.text, "Gemini (✓)"
        except Exception as e:
            st.error(f"Gemini Error: {str(e)}")
    
    # الـ Fallback في حالة فشل Gemini أو عدم وجود مفتاح
    st.warning("Switching to HuggingFace fallback...")
    hf_response = query_huggingface(user_input, hf_token)
    return hf_response, "HuggingFace (Fallback)"

# 6. عرض المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. مدخلات المستخدم
if not api_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to start.")
else:
    if user_input := st.chat_input("How can I help you today?"):
        # إضافة رسالة المستخدم للتايخ والعرض
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # جلب رد الذكاء الاصطناعي
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                full_response, model_used = get_ai_response(user_input, api_key, hf_token)
                st.markdown(full_response)
                st.caption(f"Source: {model_used}")
        
        # إضافة رد المساعد للتاريخ
        st.session_state.messages.append({"role": "assistant", "content": full_response})
