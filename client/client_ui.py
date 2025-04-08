import streamlit as st
from uuid import uuid4
import db
import requests

st.set_page_config(layout="wide")
db.init_db()

# --- Sidebar: Chat History ---
st.sidebar.title("💬 Chat History")
sessions_with_titles = db.get_sessions_with_titles()
session_labels = [title for _, title in sessions_with_titles]
session_ids = [sid for sid, _ in sessions_with_titles]

# Add option to create a new chat
selected_label = st.sidebar.selectbox("Select a chat", session_labels + ["➕ New Chat"])
if selected_label == "➕ New Chat":
    selected_session = str(uuid4())
    new_title = st.sidebar.text_input(
        "Enter a title for this chat", key="new_chat_title"
    )
    if not new_title:
        st.stop()  # Wait until title is provided
else:
    selected_session = session_ids[session_labels.index(selected_label)]
    new_title = None  # Already existing chat

st.sidebar.markdown("---")

# --- Main Chat Area ---
st.title("📚 RAG Chatbot with Documents")
chat_container = st.container()

# Load previous messages
for question, answer, docs in db.get_session_messages(selected_session):
    with chat_container:
        st.chat_message("user").markdown(question)
        st.chat_message("assistant").markdown(answer)
        if docs:
            with st.expander("📄 Referenced Documents"):
                for doc in docs:
                    st.markdown(
                        f"""
                        **📘 Document:** `{doc.get('document_name')}`  
                        - Page: `{doc.get('page_number')}`  
                        - Similarity: `{round(doc.get('similarity_score', 3))}`  
                        - Chunk: `{doc.get('chunk_text')[:250]}...`
                        """
                    )

# Clear history button
if st.sidebar.button("🧹 Clear History"):
    db.clear_history()
    st.success("Chat history cleared! Please refresh the page.")

# --- Chat Input ---
query = st.chat_input("Ask a question based on uploaded documents...")
if query:
    with st.spinner("💬 Generating answer..."):
        response = requests.post(
            "http://localhost:8000/query",  # Your FastAPI endpoint
            json={"input": query, "top_k": 10},
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "No answer found.")
            docs = data.get("vector_results", [])

            # Save to DB
            db.save_message(selected_session, query, answer, docs, title=new_title)

            # Show in UI
            st.chat_message("user").markdown(query)
            st.chat_message("assistant").markdown(answer)
            if docs:
                with st.expander("📄 Referenced Documents"):
                    for doc in docs:
                        st.markdown(
                            f"""
                            **📘 Document:** `{doc.get('document_name')}`  
                            - Page: `{doc.get('page_number')}`  
                            - Similarity: `{round(doc.get('similarity_score', 3))}`  
                            - Chunk: `{doc.get('chunk_text')[:250]}...`
                            """
                        )
        else:
            st.error("❌ Failed to get response from API.")
