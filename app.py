import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

client = Groq(api_key=api_key)

# System prompt — this defines your chatbot's personality
SYSTEM_PROMPT = """
You are an expert Career Coach with 20 years of experience helping 
professionals land their dream jobs. You help with:
- Resume writing and optimization
- Interview preparation and mock interviews
- Career path guidance
- LinkedIn profile improvement
- Salary negotiation tips

Be friendly, encouraging, and give specific, actionable advice.
Keep responses concise and practical.
"""

# Page config
st.set_page_config(page_title="Career Coach AI", page_icon="💼")
st.title("💼 Career Coach AI")
st.caption("Your personal AI-powered career advisor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your career..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for API
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    api_messages += st.session_state.messages

    # Get response from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                    max_tokens=1000
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
            except Exception as e:
                reply = "I'm sorry, I encountered an error processing your request."
                st.error(f"API Error: {e}")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": reply})