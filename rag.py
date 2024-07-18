import streamlit as st
import os
import time
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set the Groq API key
os.environ["GROQ_API_KEY"] = "an API here"

# Initialize session state
def initialize_session_state():
    if 'template' not in st.session_state:
        st.session_state.template = """You are a helpful chatbot.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:
        
        """

    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template,
        )

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = Chroma(
            persist_directory='emb',
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# Get response from Groq API
def get_groq_response(client, user_query, context):
    sys_prompt = f"""
    Instructions:
    - Be a helpful chatbot 
    Context: {context}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query}
        ],
        model="mixtral-8x7b-32768"
    )
    return chat_completion.choices[0].message.content

# Main function
def main():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    initialize_session_state()

    st.title("RAG")

    # Uploading a PDF file
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

    if uploaded_file is not None:
        fp = f'The_files/{uploaded_file.name}'
        
        # Check if the file already exists
        if not os.path.isfile(fp):
            with st.spinner("Analyzing your document..."):
                bytes_data = uploaded_file.read()
                
                # Save the uploaded file
                with open(fp, "wb") as f:
                    f.write(bytes_data)
                
                # Load the PDF
                loader = PyPDFLoader(fp)
                data = loader.load()

                # Initialize TokenTextSplitter
                text_splitter = TokenTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200
                )
                all_splits = text_splitter.split_documents(data)

                if not all_splits:
                    st.error("The document couldn't be processed. It might be empty or too short.")
                    return

                try:
                    # Create and persist the vector store
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    )
                    st.session_state.vectorstore.persist()
                except Exception as e:
                    st.error(f"An error occurred while processing the document: {str(e)}")
                    return

        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

        # Chat input
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    try:
                        # Retrieve relevant context from the vector store
                        context_documents = st.session_state.retriever.get_relevant_documents(user_input)
                        context = " ".join([doc.page_content for doc in context_documents])

                        response = get_groq_response(client, user_input, context)
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                        chatbot_message = {"role": "assistant", "message": response}
                        st.session_state.chat_history.append(chatbot_message)
                    except Exception as e:
                        st.error(f"An error occurred while processing your request: {str(e)}")
    else:
        st.write("Please upload a PDF file.")

if __name__ == "__main__":
    main()
