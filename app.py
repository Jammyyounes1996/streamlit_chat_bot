import streamlit as st
import google.generativeai as genai
import requests

# إعدادات الصفحة
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("🤖 AI Chatbot")

# الإعدادات في الشريط الجانبي
with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    hf_token = st.text_input("HuggingFace Token (Optional):", type="password")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# دالة HuggingFace برابط الـ Inference API الصحيح
def query_huggingface(prompt, hf_token):
    try:
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        # تم تغيير الرابط للصيغة الأكثر استقراراً
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
        payload = {"inputs": f"<s>[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 250}}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            return f"HF Error {response.status_code}: Please check token or model status."
    except Exception as e:
        return f"HF Exception: {str(e)}"

# دالة الاستجابة الرئيسية
def get_ai_response(user_input, api_key, hf_token):
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # تجربة استدعاء الموديل بدون كلمة models/ وبدون إضافات لتقليل احتمالية الخطأ 404
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)
            return response.text, "Gemini"
        except Exception as e:
            # محاولة أخيرة باستخدام gemini-pro إذا فشل flash
            try:
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(user_input)
                return response.text, "Gemini Pro"
            except:
                st.error(f"Gemini API Error: {str(e)}")
    
    st.warning("Switching to Fallback...")
    return query_huggingface(user_input, hf_token), "HuggingFace"

# عرض الشات
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Write here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Please provide an API Key first.")
        else:
            res, source = get_ai_response(user_input, api_key, hf_token)
            st.write(res)
            st.caption(f"Source: {source}")
            st.session_state.messages.append({"role": "assistant", "content": res})
