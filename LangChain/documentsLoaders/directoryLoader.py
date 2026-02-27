from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    "researchPapers",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
