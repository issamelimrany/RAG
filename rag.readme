# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit, Groq, and various NLP libraries. The chatbot can process PDF documents, answer questions based on the content, and maintain a conversation history.

## Features

- PDF document upload and processing
- Text embedding and vector store creation
- Contextual question answering using RAG
- Real-time chat interface with Streamlit
- Integration with Groq API for language model inference

## Prerequisites

- Python 3.7+
- Groq API key

## Installation

1. Clone this repository:
   ```
   git clone the repo
   cd to it
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Groq API key:
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Upload a PDF document using the file uploader.

4. Once the document is processed, you can start chatting with the bot about the content of the PDF.

## How it Works

1. **Document Processing**: The app uses PyPDFLoader to read PDF files and TokenTextSplitter to split the text into manageable chunks.

2. **Embedding and Indexing**: Text chunks are embedded using HuggingFace's sentence transformers and stored in a Chroma vector store.

3. **Query Processing**: When a user asks a question, the app retrieves relevant context from the vector store.

4. **Response Generation**: The retrieved context and user query are sent to the Groq API, which generates a response using the Mixtral 8x7B model.

5. **Chat Interface**: Streamlit provides a user-friendly chat interface for interacting with the bot.

## Customization

- You can modify the `template` in the `initialize_session_state()` function to change the chatbot's behavior.
- Adjust the `chunk_size` and `chunk_overlap` in the `TokenTextSplitter` to optimize document processing for your needs.
- Experiment with different embedding models by changing the `model_name` in the `HuggingFaceEmbeddings` initialization.

