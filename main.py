from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

# provide the path of  pdf file/files.
pdfreader = PdfReader('budget_speech.pdf')


from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


raw_text


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


len(texts)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


document_search = FAISS.from_texts(texts, embeddings)

document_search

chain = load_qa_chain(OpenAI(), chain_type="stuff")


query = "Vision for Amrit Kaal"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)


query = "How much the agriculture target will be increased to and what the focus will be"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)