# streamlit_app.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from groq import Groq  
from dotenv import load_dotenv

load_dotenv(override=True)

# ----- Setup -----
st.set_page_config(page_title="Support Chatbot", page_icon="🤖")
st.title("📘 AngelOne Insurance RAG Chatbot")
st.markdown("🤖 *Ask me anything about AngelOne support or your insurance plans!* I’ve read the docs so you don’t have to.")

# ----- Load VectorDB -----
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vector_db()

# ----- Ask Groq LLaMA -----
def query_groq_llama(question, top_docs):

    # Prepare the context from top documents
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # Reconstruct conversation history from session state
    history_block = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.get("chat_history", [])])

    # Final prompt with history + context
    prompt = f"""
                You are a helpful support assistant.
                Only answer the user's question using the provided context. 
                If the answer is not in the context, respond with "I am Sorry I am not aware of that, I can only assist you with AngleOne related queries.".

                ### Conversation History ###
                {history_block}

                ### Question ###
                {question}

                ### Context ###
                {context}

                ### Important Instruction ###
                Only return the Answer to the question. Do not attach any extra text to it.
                """

    client = Groq(
    # This is the default and can be omitted
        api_key="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"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AngelOne Customer Support Specialist. Your primary goal is to deliver timely and accurate assistance to our valued customers, ensuring their experience is seamless and positive. You approach every query with a customer-first mindset, actively listening to understand their needs, offering clear and concise solutions, and guiding them step by step through any challenges they may face. You remain calm, friendly, and approachable, making sure that each interaction is not just about resolving an issue but also about building trust and satisfaction. Always offer the best possible advice and escalate issues when necessary to ensure optimal support."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    response= chat_completion.choices[0].message.content
    print("Response: ", response)

    return response


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----- UI -----
st.divider()
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("Your Question:", key="query_input")
    submitted = st.form_submit_button("Send")

if submitted and query:
    with st.spinner("Thinking..."):
        docs = vectorstore.similarity_search(query, k=3)
        answer = query_groq_llama(query, docs)
        st.session_state.chat_history.append((query, answer))
        st.markdown(f"**Answer:**\n\n{answer}")

# Show conversation
if st.session_state.chat_history:
    st.subheader("🗂️ Conversation History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**User:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")



# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.chat_history = []  # Clear the chat history
    st.text_input("Your Question: ", value="", key="reset_query")  # Reset input field
    st.rerun()  # Rerun the app to update UI

