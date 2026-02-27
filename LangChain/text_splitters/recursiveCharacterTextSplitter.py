from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

loader_docs = PyPDFLoader('Attention (Research Paper).pdf')
loader_text = TextLoader('story.txt', encoding="utf-8")

docs = loader_docs.load()
text = loader_text.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

# chunks = splitter.split_text(loader_text)

chunks_doc = splitter.split_documents(text)
print(len(chunks_doc))

print(chunks_doc[0])