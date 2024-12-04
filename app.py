import os
from dotenv import dotenv_values
import streamlit as st
from groq import Groq


def parse_groq_stream(stream):
    # Yield content chunk by chunk as the stream arrives
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# streamlit page configuration
st.set_page_config(
    page_title="TMS Chatbot Careforce",
    layout="centered",
)

# Load environment variables
try:
    secrets = dotenv_values(".env")  # for dev env
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except:
    secrets = st.secrets  # for streamlit deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# save the api_key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INITIAL_RESPONSE = secrets["INITIAL_RESPONSE"]
INITIAL_MSG = secrets["INITIAL_MSG"]
CHAT_CONTEXT = secrets["CHAT_CONTEXT"]

client = Groq()

# initialize the chat history if not present in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": INITIAL_RESPONSE},
    ]

# page title
st.title("Welcome to the Chatbot!")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_prompt = st.chat_input("Ask me")

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": CHAT_CONTEXT},
        {"role": "assistant", "content": INITIAL_MSG},
        *st.session_state.chat_history
    ]

    # Display assistant response as it streams
    with st.chat_message("assistant"):
        # Create the streaming request to Groq API
        stream = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            stream=True  # Stream the response
        )
        
        # Accumulate the full response in one go, then display it
        full_response = ""
        for chunk in parse_groq_stream(stream):
            full_response += chunk  # Accumulate all the response

        # After accumulating the full response, display it
        st.markdown(full_response)  

    # Add the final response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
