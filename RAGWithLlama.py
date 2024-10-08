#---------SETUP---------#
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


#---------DOCUMENT LOADING---------#
#lading all 27 articles from ISAW Papers (static version)
loaders = [WebBaseLoader(f"https://dlib.nyu.edu/awdl/isaw/isaw-papers/{x}/") for x in range(1, 28)]
len(loaders)

docs = []
for loader in loaders:
    docs.extend(loader.load())


#---------DOCUMENT SPLITTING---------#
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
len(splits)


#---------VECTORS AND EMBEDDING---------#
!ollama pull nomic-embed-text
!ollama list

#add to vector database
vector_db = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    collection_name="local-rag"
)


#---------QUESTIONS AND ANSWERS---------#
local_model = "llama2"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI digital assistant. Your task is to generate five different versions of the given user question to retrieve relevant information from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)
 
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

#RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
if you don't know the answer just say so. 
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutParser()
)

chain.invoke("is there anything about Athens?")
