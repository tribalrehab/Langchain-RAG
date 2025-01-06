from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

if os.path.isfile("./chroma_db/chroma.sqlite3"):
    # Load vector database
    print("Loading vector database")
    # load from disk
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    print("Done loading vector database")
else:
    print("Creating vector database")
    print("Importing documents")
    loader = TextLoader('article.txt')
    documents = loader.load()
    print("Import completed")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(),persist_directory="./chroma_db")
    print("Vector database created")

local_path = (
    "./gpt4all-falcon-q4_0.gguf"
)
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

query = "tell me about the Core Ultra 200H"
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),verbose=True,chain_type_kwargs={"verbose": True})
print(query)
print(qa)
qa.run(query)
