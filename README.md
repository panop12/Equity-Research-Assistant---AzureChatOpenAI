# Equity-Research-Assistant---AzureChatOpenAI
This project implements an equity research assistant using LangChain, Streamlit, and OpenAI's GPT models. The assistant is designed to analyze and generate detailed financial reports based on user queries and provided PDF documents. Users can upload PDFs containing financial reports or other relevant documents, and the assistant will process these files to provide insights and recommendations.

Features
PDF Document Processing: Upload multiple PDF documents for analysis.
Text Extraction: Extract text from the uploaded PDFs.
Text Chunking: Split extracted text into manageable chunks for processing.
Vector Store Creation: Create a vector store from text chunks using OpenAI embeddings and FAISS.
Conversational Interface: Engage in a conversation with the assistant, which retains context using a conversational retrieval chain.
Customizable Prompts: Generate prompts for equity research analysis based on user questions and context.
Technologies Used
LangChain: For creating the language model chains and handling text processing.
Streamlit: For building the interactive web interface.
PyPDF2: For extracting text from PDF files.
FAISS: For creating the vector store from text embeddings.
OpenAI: For generating embeddings and conversational responses.
Azure OpenAI: For deploying and managing the OpenAI models.
dotenv: For loading environment variables.
Getting Started
Prerequisites
Python 3.7 or higher
OpenAI API key and Azure OpenAI deployment setup
Required Python packages (listed in requirements.txt)
