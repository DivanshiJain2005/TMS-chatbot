import os
import streamlit as st
from groq import Groq

# Function to parse Groq stream response
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# Streamlit page configuration
st.set_page_config(
    page_title="Transcranial Magnetic Stimulation Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Load secrets from Streamlit Cloud
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    INITIAL_RESPONSE = st.secrets["INITIAL_RESPONSE"]
    INITIAL_MSG = st.secrets["INITIAL_MSG"]
    CHAT_CONTEXT = st.secrets["CHAT_CONTEXT"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please ensure the required secrets are set in Streamlit Cloud.")

# Set the API key in the environment variable (good practice for external API calls)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client
client = Groq()

# Initialize the chat history if not already in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": INITIAL_RESPONSE},
    ]

# Page title
st.title("Welcome to the bot!")
st.caption("Helping You Level Up Your TMS knowledge")

# Display chat history (messages)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar='🤖' if message["role"] == "assistant" else '🗨️'):
        st.markdown(message["content"])

# User input field
user_prompt = st.chat_input("Ask me")

if user_prompt:
    # Add user message to chat history
    with st.chat_message("user", avatar="🗨️"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare messages for the Groq model
    messages = [
        {"role": "system", "content": CHAT_CONTEXT},
        {"role": "assistant", "content": INITIAL_MSG},
        *st.session_state.chat_history
    ]

    # Display assistant response in the chat message container
    with st.chat_message("assistant", avatar='🤖'):
        # Request completion from Groq API with streaming enabled
        stream = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            stream=True  # Streaming mode
        )

        # Accumulate the response from the stream
        full_response = ""
        for chunk in parse_groq_stream(stream):
            full_response += chunk  # Concatenate each chunk

        # After full response is accumulated, display the complete response
        st.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
