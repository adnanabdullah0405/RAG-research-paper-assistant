import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
from dotenv import load_dotenv
from gtts import gTTS  # Import Google Text-to-Speech
from sentence_transformers import SentenceTransformer
from langchain.prompts.chat import ChatPromptTemplate
import io  # Import io for in-memory file operations

# Load environment variables
load_dotenv()

# Access environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

# Streamlit page configuration
st.set_page_config(page_title="GyroGlove AI Assistant", layout="wide")
st.title("🤖 GyroGlove AI Assistant for Hand Tremor")

# Memory to store conversation
memory = []  # List to store previous questions and answers

# Sidebar: Memory options and GyroGlove image
with st.sidebar:
    st.image("gyroglove.jpeg", width=300)
    st.header("🔧 Features")

    if st.button("🧹 Clear Memory"):
        memory.clear()
        st.write("Memory Cleared!")
    
    st.markdown("<h3 style='color: #007BFF;'>Frequently Asked Questions:</h3>", unsafe_allow_html=True)
    
    def handle_faq_click(question):
        return question

    if st.button("What is GyroGlove?", key="faq_gyroglove"):
        user_question = handle_faq_click("What is GyroGlove?")
    
    if st.button("How does GyroGlove work?", key="faq_work"):
        user_question = handle_faq_click("How does GyroGlove work?")
    
    if st.button("What is a tremor?", key="faq_tremor"):
        user_question = handle_faq_click("What is a tremor?")
    
    if st.button("How do tremors affect daily life?", key="faq_life"):
        user_question = handle_faq_click("How do tremors affect daily life?")
    
    if st.button("What are common medications for tremors?", key="faq_medications"):
        user_question = handle_faq_click("What are common medications for tremors?")
    
    st.markdown("<h3 style='color: #007BFF;'>Feedback:</h3>", unsafe_allow_html=True)
    
    # Feedback buttons with responses
    feedback = st.radio("How was your experience?", ["👍 Good Work", "👎 Needs Improvement"])
    
    if feedback == "👍 Good Work":
        st.success("Thank you for your feedback! We're glad you're satisfied! 😊")
    elif feedback == "👎 Needs Improvement":
        st.warning("Thank you for your feedback! We’ll work to improve! 🙏")

    st.markdown("<h3 style='color: #007BFF;'>Resources:</h3>", unsafe_allow_html=True)
    if st.button("📚 View Article on Hand Tremors"):
        st.markdown("[Read the Article](https://www.sciencedirect.com/topics/medicine-and-dentistry/hand-tremor)", unsafe_allow_html=True)

    if st.button("🎥 Watch Hand Tremor Video"):
        st.video("https://www.youtube.com/watch?v=2xihXzJNd8Y")

# Step 1: Load the Hugging Face model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your chosen Hugging Face model

# Step 2: Create a wrapper for the embedding function that has an `embed_query` method
class HuggingFaceEmbeddingWrapper:
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    
    def embed_documents(self, texts):
        return [self.model.encode([text])[0].tolist() for text in texts]

# Initialize the embedding wrapper
embedding_wrapper = HuggingFaceEmbeddingWrapper(model)

# Step 3: Initialize the Cassandra session using cassio
session = cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID
)

# Initialize Cassandra VectorStore for querying using the embeddings from Hugging Face
astra_vector_store = Cassandra(
    embedding=embedding_wrapper,
    table_name="qa_mini_demo",
    session=session,
    keyspace=None  # Replace with your actual keyspace
)

# Wrap the vector store for querying purposes
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Step 4: Initialize Gemini (Google Generative AI) LLM for Chat
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Choose "gemini-1.5-flash" or another model if needed
    temperature=0.7,  # Control randomness of the response
    max_tokens=512,   # Set a limit for the number of tokens in the response
    max_retries=2,    # Number of retries in case of failure
    timeout=None,     # Timeout for the API call
)

# Function to retrieve relevant text from the PDF stored in Astra DB
def get_relevant_text_from_pdf(question: str):
    relevant_docs = astra_vector_store.similarity_search(question, k=3)
    if len(relevant_docs) == 0:
        st.warning("No relevant documents were retrieved from Astra DB.")
        return ""
    
    combined_context = " ".join([doc.page_content for doc in relevant_docs])
    return combined_context

# Function to check for questions about the creator
def check_for_creator_question(question: str):
    if "who is your creator" in question.lower() or "who is your owner" in question.lower():
        return (
            "Hello! I was created by **Muhammad Adnan**, a passionate Electrical Engineering "
            "student at **NUST** (7th semester). His **Final Year Project (FYP)** is to design a revolutionary "
            "'**GyroGlove**' aimed at reducing hand tremors. Inspired by this groundbreaking work, "
            "Adnan built me to assist with knowledge about the GyroGlove and hand tremors."
        )
    return None

# Create a prompt template for the LLM to generate answers based on the context and memory
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that remembers previous conversations. Use the following memory: {memory}. "
            "Here is the context: {context}. Answer the user's question.",
        ),
        ("human", "{question}"),
    ]
)

# Combine LLM and prompt template for querying (Define the chain)
chain = prompt_template | llm

# Function to clean the response and remove metadata
def clean_response(response):
    return response.content.replace('\n', ' ').strip()

# Function to generate speech using gTTS and stream it directly in Streamlit
def generate_audio(text):
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()  # Create an in-memory bytes buffer
    tts.write_to_fp(audio_buffer)  # Write audio to buffer
    audio_buffer.seek(0)  # Move to the start of the buffer
    st.audio(audio_buffer, format="audio/mp3")  # Play audio directly from buffer

# Function to interact with the LLM, taking memory into account
def ask_question(question):
    memory.clear()
    creator_response = check_for_creator_question(question)
    if creator_response:
        return creator_response
    context_from_pdf = get_relevant_text_from_pdf(question)
    if not context_from_pdf:
        return "No context retrieved from the PDF."
    memory_context = " ".join(memory)
    response = chain.invoke({
        "memory": memory_context,
        "context": context_from_pdf,
        "question": question
    })
    cleaned_answer = clean_response(response)
    memory.append(f"Q: {question}")
    memory.append(f"A: {cleaned_answer}")
    return cleaned_answer

# Main section for chat and AI interaction
st.subheader("Ask the AI about GyroGlove or Hand Tremors")

# Input for user question
user_question = st.text_input("Enter your question:")

# Button to ask the AI
if st.button("🧠 Ask the AI"):
    if user_question:
        answer = ask_question(user_question)
        st.write(f"**Answer:** {answer}")
        generate_audio(answer)  # Generate audio without saving it
    else:
        st.warning("Please enter a question before asking the AI.")

# Footer with some info
st.markdown("---")
st.markdown("This AI-powered assistant provides insights on GyroGlove technology and its role in reducing hand tremors.")
