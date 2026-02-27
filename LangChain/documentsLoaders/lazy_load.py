from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    "researchPapers",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

for page in loader.lazy_load():
    # print(page.page_content)
    print(page.metadata)

