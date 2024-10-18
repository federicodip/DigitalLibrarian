#!/usr/bin/env python
# coding: utf-8

# In[15]:


from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
import os 
import json
import time
import aiofiles
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import openai
import sys
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# In[8]:


openai.api_key = os.getenv("OPENAI_API_KEY")


# In[9]:


os.environ["USER_AGENT"] = "QAGeneratorWeb"
from langchain.document_loaders import WebBaseLoader


# In[10]:


#extract list of all 57 ISAW article 
file_url = "isaw_url.txt" 

# Open the file and read the URLs
with open(file_url, 'r') as file:
    article_list = [line.strip() for line in file]

# Output the list to verify
len(article_list)


# In[11]:


def file_loading_splitting(file_path): 
    loader = WebBaseLoader(file_path)
    data = loader.load()

    splitter_ques = TokenTextSplitter(model_name = 'gpt-3.5-turbo', chunk_size = 10000, chunk_overlap = 200 )
    splitter_ans = TokenTextSplitter(model_name ='gpt-3.5-turbo', chunk_size = 1000, chunk_overlap=100)

    question_gen = ''.join([page.page_content for page in data])

    chunks_ques = splitter_ques.split_text(question_gen)
    document_ques = [Document(page_content=t) for t in chunks_ques]

    document_ans = splitter_ans.split_documents(document_ques)

    return document_ques, document_ans


# In[14]:


#Retrieve article title
def get_article_title(url):
    # Fetch the content of the webpage
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the <title> tag (basic method)
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
        
        # Alternatively, try to find a meta tag with name="DC.title"
        meta_title = soup.find('meta', attrs={"name": "DC.title"})
        if meta_title and 'content' in meta_title.attrs:
            return meta_title['content']
    
    return None


# In[22]:


# Retrieve article metadata
def get_article_metadata(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract title from the h1 tag with property 'dcterms:title'
    title_element = soup.find('h1', {'property': 'dcterms:title'})
    title = get_article_title(url)

    # Extract authors from the h2 tag with rel 'dcterms:creator'
    authors = []
    author_h2 = soup.find('h2', {'rel': 'dcterms:creator'})
    if author_h2:
        # Find all span tags with property='foaf:name' within the h2 tag
        author_spans = author_h2.find_all('span', {'property': 'foaf:name'})
        if author_spans:
            for span in author_spans:
                author_name = span.get_text().strip()
                authors.append(author_name)
        else:
            # If no spans are found, get text directly from h2
            author_name = author_h2.get_text().strip()
            authors.append(author_name)
    author = ', '.join(authors) if authors else ''

    # Extract volume information
    volume_element = soup.find('h2')
    volume = volume_element.get_text().strip().split()[-2] if volume_element else ''

    # Extract year information from the footer
    footer_element = soup.find('footer')
    if footer_element:
        footer_text = footer_element.find('p').get_text()
        year = footer_text.split("©")[1].split()[0].strip() if "©" in footer_text else ''
    else:
        year = ''

    # Extract the Permanent URL
    url_element = soup.find('link', {'rel': 'canonical'})
    permanent_url = url_element['href'].strip() if url_element else ''

    return {
        'title': title,
        'author': author,
        'volume': volume,
        'year': year,
        'url': permanent_url
    }

def generate_citation(url):
    if url == 'https://dlib.nyu.edu/awdl/isaw/isaw-papers/15/' or url == 'https://dlib.nyu.edu/awdl/isaw/isaw-papers/17/':
        return get_article_title(url)
    if url in article_list[6:37]:
        metadata = get_article_metadata(url)
        citation = f"{metadata['author']} {metadata['year']},\"{metadata['title']}\" ISAW Papers, vol. {metadata['volume']}, {metadata['url']}."
    else:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        citation_element = soup.find('meta', {'name': 'DCTERMS.bibliographicCitation'})
        if citation_element:
            citation = citation_element['content'].strip()
        else:
            citation = soup.find('title').get_text().strip()
    return citation
    
for x in article_list:
    print(generate_citation(x))


# In[23]:


def llm_pipeline(file_path):

    # Process the document and generate questions
    document_ques_gen, document_answer_gen = file_loading_splitting(file_path)
    # Initialize output Q&A pair list 
    qa_pairs = []

    # select LLM for Q&A generation
    llm_ques_gen = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # select vector store and embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Prompt templates for question generation and refinement 
    prompt_template = """
    You are an expert at creating questions based on scholarly articles and essays.
    Your goal is to prepare students or scholars for their research work.\n",
    You do this by asking questions about the text below:
    ------------
    {text}
    ------------
    Create questions that will inform researchers about the content of the texts.
    Make sure not to lose any important information.
    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at creating questions based on scholarly articles and essays.
    Your goal is to prepare students or scholars for their research work.
    We have received some questions so far: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------
    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )


    # Initialize question generation chain
    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen, 
                                          chain_type="refine", 
                                          verbose=False, 
                                          question_prompt=PROMPT_QUESTIONS, 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Run question generation getting in input the preprocessed Document file
    ques = ques_gen_chain.run(document_ques_gen)

    # Splits the questions and adds them to a list of questions
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    # Initialize answer generation  
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                          chain_type="stuff", 
                                                          retriever=vector_store.as_retriever())
    
    # generate answers and store question-answer pairs with sources
    for question in filtered_ques_list:
        print("Generating answer for question:", question)
        answer = answer_generation_chain.run(question)
        
        # Call the source generation function
        
        source_text = generate_citation(file_path)  # Assume you have this function
        
        # Append the answer with the source text
        qa_pairs.append({
            "question": question, 
            "answer": answer,
            "source": source_text  # Append the source to the answer
        })

    return qa_pairs
#llm_pipeline('https://dlib.nyu.edu/awdl/isaw/isaw-papers/3/')


# In[24]:


def gen_qa_and_save(file_path):

    qa_pairs = llm_pipeline(file_path)

    # Define the output file path
    output_file = "QAOpenAI.json"

    # Step 1: Check if the file exists and load existing data if it does
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as json_file:
            try:
            # Load existing data from the file
                existing_data = json.load(json_file)
            
            # Ensure it's a list; if it's not, wrap it in a list
                if isinstance(existing_data, dict):
                # If it's a dict (e.g. the first run), convert it into a list
                    existing_data = [existing_data]
            except json.JSONDecodeError:
            # If the file is empty or corrupted, initialize as an empty list
                existing_data = []
    else:
    # If the file doesn't exist, initialize as an empty list
        existing_data = []

    # Step 2: Append the new data (qa_pairs) to the existing data list
    existing_data.append({
        "title": get_article_title(file_path),
        "source": file_path,
        "qa_pairs": qa_pairs
    })

    # Step 3: Save the updated data back to the JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

    print(f"Question-Answer pairs appended and saved to {output_file}")
    return None


# In[25]:


#to generate all the Q&A pairs
'''
for url in article_list:
    qa_pairs = gen_qa_and_save(url)
'''

