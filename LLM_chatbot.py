import streamlit as st
import ollama

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Local AI Chatbot (No API Key)", layout="centered")

st.title("Local AI Chatbot (No API Key)")
st.caption("Runs on your laptop using Ollama models")

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that answers questions accurately and concisely."
        }
    ]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Application Settings")

    model_name = st.selectbox(
        "Choose a Model:",
        options=["phi3:mini", "llama3.1:8b", "mistral"],
        index=0
    )

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max Tokens (response length)", 64, 1024, 256, 64)

    st.markdown("---")
    st.subheader("Pull model first if needed")
    st.code(f"ollama pull {model_name}", language="bash")

    st.markdown("---")
    st.code("ollama list", language="bash")

# -------------------- DISPLAY CHAT HISTORY --------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- RESPONSE FUNCTION --------------------
def generate_response_stream(messages, model, temp, max_tokens):
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temp,
            "num_predict": max_tokens
        }
    )

    for chunk in stream:
        yield chunk["message"]["content"]

# -------------------- USER INPUT --------------------
user_text = st.chat_input("Please type your question here...")

if user_text:
    st.session_state.messages.append(
        {"role": "user", "content": user_text}
    )

    with st.chat_message("user"):
        st.markdown(user_text)

    # Assistant response
    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                generate_response_stream(
                    messages=st.session_state.messages,
                    model=model_name,
                    temp=temperature,
                    max_tokens=max_tokens
                )
            )

        except Exception as e:
            st.error("Could not connect to Ollama or run the selected model.")
            st.exception(e)
            response_text = ""

    if response_text:
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

# -------------------- CLEAR CHAT --------------------
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions accurately and concisely."
            }
        ]
        st.rerun()