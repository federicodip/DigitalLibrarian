#ISAW Papers Digital Librarian: A Retrieval-Augmented Generation (RAG) Chatbot

This Python application allows users to interact with a collection of scholarly articles from the ISAW Papers (Institute for the Study of the Ancient World) through a Retrieval-Augmented Generation (RAG) chatbot. The application combines document retrieval and language generation using OpenAI's GPT model.

Key Features:
Document Retrieval: Loads and indexes all 27 ISAW Papers using the Chroma vector store and OpenAI embeddings.
Question Answering: Provides answers based on relevant documents retrieved through semantic search.
User Interface: Includes a Gradio-powered web interface where users can ask questions or submit research abstracts to get responses from the RAG model.
Alternative Interface: Also includes an optional Tkinter-based desktop UI for local interaction.
