from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load pdf
loader = PyPDFLoader("./현진건_운수좋은날.pdf")
documents = loader.load_and_split()

# Split text (Since it's too long, split it into 1000 pieces each. overlap 200 for the naturalness of the sentence)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 100)
texts = text_splitter.split_documents(documents)

# Chroma DB
persist_directory = 'db_pdf' # Save to directory named 'db_pdf'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding = embedding, # Which embedding
    persist_directory = persist_directory)

# Initialization
vectordb.persist()
vectordb = None

# Database, Load saved data into memory(RAM)
vectordb = Chroma(
    persist_directory = persist_directory,
    embedding_function = embedding
)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0, streaming=True),
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def main():
    while True:
        query = input('Question about your pdf\n type "q" if you want to quit: ')
        if query == "q":
            break
        try:
            llm_response = qa_chain(query)
            print(llm_response["result"],'\n')
        except:
            print("ERROR")
            break

if __name__ == "__main__":
    main()