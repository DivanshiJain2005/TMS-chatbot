import os
import json
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to parse Groq stream response
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# Streamlit page configuration
st.set_page_config(
    page_title="TMS Chatbot Careforce",
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

# Initialize Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)

# Load data from the 'data.json' file (replace with your actual file path)
def load_data():
    with open('data.json', 'r', encoding="utf8") as file:
        return json.load(file)

# Perform a simple keyword-based search using TF-IDF
def search_data(query, data):
    # Flatten the data into a list of titles and contents
    data_text = [f"{entry['title']} {entry['content']}" for entry in data["documents"]]
    
    # Combine query with data for search
    documents = [query] + data_text

    # Convert the documents to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between the query and the documents
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    most_similar_index = similarity_scores.argmax()

    # Return the most relevant document based on the similarity score
    return data["documents"][most_similar_index]

# Initialize the chat history if not already in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": INITIAL_RESPONSE},
    ]

# Page title
st.title("Welcome to Careforce!")

# Display chat history (messages)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_prompt = st.chat_input("Ask me")

if user_prompt:
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Load the TMS data from the JSON file
    tms_data = load_data()

    # Search for relevant information from the TMS data based on the user query
    relevant_data = search_data(user_prompt, tms_data)
    relevant_info = relevant_data["content"]  # Use the content of the most relevant document

    # Prepare messages for the Groq model
    messages = [
        {"role": "system", "content": CHAT_CONTEXT},
        {"role": "assistant", "content": INITIAL_MSG},
        {"role": "assistant", "content": "Here is some relevant information:"},
        {"role": "assistant", "content": relevant_info},
        *st.session_state.chat_history
    ]

    # Display assistant response in the chat message container
    with st.chat_message("assistant"):
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
