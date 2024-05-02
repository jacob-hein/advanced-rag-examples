import fandom
import os

from fandom import FandomPage
from mdutils.mdutils import MdUtils

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter


def get_malazan_pages(articles=["Anomander Rake", "Tayschrenn", "Kurald Galain", "Warrens", "Tattersail", "Whiskeyjack", "Kruppe"]) -> list[FandomPage]:
    """
    Retrieves FandomPage objects for the specified articles from the Malazan wiki.

    Args:
        articles (list[str], optional): A list of article names to retrieve. Defaults to ["Anomander Rake", "Tayschrenn", "Kurald Galain"].

    Returns:
        list[FandomPage]: A list of FandomPage objects corresponding to the specified articles.
    """
    fandom.set_wiki("malazan")
    pages = [fandom.page(article) for article in articles]
    return pages

def create_and_save_md_files(pages, path="./data/docs/"):
    """
    Create Markdown files based on the given page objects.

    Args:
        pages (list[FandomPage]): A list of page objects containing the content to be written to the Markdown files.

    Returns:
        None
    """
    for page in pages:
        create_and_save_md_file(page, path)

def create_and_save_md_file(page, path="./data/docs/"):
    """
    Create a Markdown file based on the given page object.

    Args:
        page (Page): The page object containing the content to be written to the Markdown file.

    Returns:
        None
    """
    title = page.content["title"]
    filename = os.path.join("", f'{path}{title.lower().replace(' ', '-')}.md')
    mdFile = MdUtils(file_name=filename, title=title)
    mdFile.new_header(level=1, title="Summary")
    mdFile.new_paragraph(page.summary)
    mdFile.new_line()
    mdFile.new_header(level=1, title=title)
    mdFile.new_paragraph(page.content["content"])
    mdFile.new_line()
    sections = page.content["sections"]
    for section in sections:
        mdFile.new_header(level=2, title=section["title"])
        mdFile.new_paragraph(section["content"])
        mdFile.new_line()
    mdFile.create_md_file()
    
def generate_vector_index(docs_path="./data/docs", chunk_size=512) -> VectorStoreIndex:
    """
    Generates a vector store index from a collection of documents.

    Args:
        docs_path (str): The path to the directory containing the documents. Defaults to "./data/docs".
        chunk_size (int): The size of each chunk for sentence splitting. Defaults to 512.

    Returns:
        VectorStoreIndex: The generated vector store index.
    """
    documents = SimpleDirectoryReader(docs_path).load_data()
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", 
        embed_batch_size=256
    )

    splitter = SentenceSplitter(chunk_size=chunk_size)
    index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter], embed_model=embed_model
    )
    return index