#---------SETUP---------#
#!pip install langchain
#pip install openai
#pip install -U langchain-community
#!pip install tiktoken
#!pip install chromadb
#!pip install lark
#pip install -U langchain-chroma
#pip install gradio

import os
import openai
import sys
openai.api_key = os.getenv("OPENAI_API_KEY")

#---------DOCUMENT LOADING---------#

from langchain.document_loaders import WebBaseLoader

#lading all 27 articles from ISAW Papers (static version)
loaders = [WebBaseLoader(f"https://dlib.nyu.edu/awdl/isaw/isaw-papers/{x}/") for x in range(1, 28)]
len(loaders)

docs = []
for loader in loaders:
    docs.extend(loader.load())

#---------DOCUMENT SPLITTING---------#
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
len(splits)

#---------VECTORSTORE AND EMBEDDING---------#
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

import numpy as np
from langchain.vectorstores import Chroma   #importing Chroma which is a lightweight verctore store
persist_directory = 'docs/chroma/'   #space to allocate date in the vectorestore
#import os
import shutil     #this cell makes sure the directory we are about to use is empty
# Define the directory path
dir_path = './docs/chroma'
# Check if the directory exists, then remove it
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

vectordb = Chroma.from_documents(    #this creates the vectorstore
    documents=splits,              #data to be stored
    embedding=embedding,       #type of embedding (from OpenAI)
    persist_directory=persist_directory    #passing the directory where the data is to be stored
)

print(vectordb._collection.count()) #it should be the same as the number of splits

#---------QUESTION ANSWERING:SIMILARITY SEARCH---------#

question = "Is there anything by Gilles Bransbourg?"

docs = vectordb.similarity_search(question,k=3)    #variabile con 3 documenti ottenuti con similarity search
#len(docs)
for d in docs:
    print(d.metadata)

from langchain.chat_models import ChatOpenAI        #importing openai as our llm for our chatbot  
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)     #setting temperature to zero to get more factual answers
from langchain.chains import RetrievalQA        #importing the type of retrival technique
qa_chain = RetrievalQA.from_chain_type(       #setting the retrival parameter
    llm,                                     #passing openai as the chosen llm
    retriever=vectordb.as_retriever()        #passing the vectordatabase that we created
)
result = qa_chain({"query": question})   #saving the result in a dictionary    
result["result"]


#---------GRADIO USER INTERFACE---------#
import gradio as gr
print(gr.__version__)

# Function to generate a response from the RAG model
def chatbot_response(question):
    result = qa_chain({"query": question})  # Use the qa_chain created in your notebook
    return result.get("result", "No response available.")

# Setting up the Gradio interface
iface = gr.Interface(
    fn=chatbot_response,  # Function that handles the input and returns output
    inputs="text",        # Input type: a single line of text
    outputs="text",       # Output type: text
    title="ISAW Papers Digital Librarian",
    description="Ask our Librarian a question about ISAW papers, or simply paste your research abstract below."
)

# Launch the Gradio app
iface.launch()


#---------ALTERNATIVE BASIC USER INTERFACE---------#
'''import tkinter as tk
from tkinter import scrolledtext
# Function to handle user input and get a response from the chatbot
def get_response():
    question = user_input.get()  # Get the user's question from the input box
    if question.strip() != "":
        # Get response from the chatbot (RAG-based retrieval)
        result = qa_chain({"query": question})  # This assumes qa_chain is already set up
        response = result.get("result", "No response")
        
        # Enable the chat history to insert new text
        chat_history.config(state='normal')
        
        # Insert user question and bot response into chat history
        chat_history.insert(tk.END, "You: " + question + "\n")
        chat_history.insert(tk.END, "Bot: " + response + "\n\n")
        
        # Scroll to the bottom of the chat history
        chat_history.yview(tk.END)
        
        # Disable the chat history to prevent direct user editing
        chat_history.config(state='disabled')
        
        # Clear the user input box for the next question
        user_input.delete(0, tk.END)
# Initialize the Tkinter window
root = tk.Tk()
root.title("RAG-based Chatbot")
root.geometry("600x400")
# Create a chat history text box (read-only)
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, state='disabled')
chat_history.grid(column=0, row=0, padx=10, pady=10)
# Create an input box for the user to type their question
user_input = tk.Entry(root, width=50)
user_input.grid(column=0, row=1, padx=10, pady=10)
# Create a submit button that will trigger the chatbot response
submit_button = tk.Button(root, text="Submit your question", command=get_response)
submit_button.grid(column=0, row=2, padx=10, pady=10)
# Run the Tkinter event loop
root.mainloop()
'''
