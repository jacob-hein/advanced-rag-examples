from openai import AzureOpenAI
import streamlit as st
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from chromadb.config import Settings
from llama_index.core import PromptTemplate

st.title("VtDat Chatbot")

client = AzureOpenAI(api_key=st.secrets["OPENAI_API_KEY"], 
                api_version="2024-05-01-preview", 
                azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"])

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002",
    api_type="azure",
    api_version="2024-05-01-preview"
)

chroma_client_load = chromadb.PersistentClient(
    path="./data/baseline-rag-pdf-docs/chromadb",
    settings=Settings(allow_reset=True)
)

# Get the existing collection by name
collection_load = chroma_client_load.get_collection(name="vtdat", embedding_function=openai_ef)
###

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the default prompt template
default_prompt = """You are a helpful assistant that answers questions about the course material from "Philosophy of Computer Science (VtDat)" using provided context.
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query. Always provide an answer in the Danish language.
    Query: {query}
    Answer: 
    """

# Add a text input in the sidebar to configure the prompt template
prompt_template = st.sidebar.text_area("Prompt Template", default_prompt)


if query := st.chat_input("What is up?"):
    #RAG
    result = collection_load.query(query_texts=[query], n_results=5)
    context = result["documents"][0]
    prompt = PromptTemplate(
    """You are a helpful assistant that answers questions about the course material from "Philosophy of Computer Science (VtDat)" using provided context.
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query. Always provide an answer in the Danish language.
    Query: {query}
    Answer: 
    """,
    )
    prompt = PromptTemplate(prompt_template)
    message = prompt.format(query=query, context="\n\n".join(context))
    ###

    st.session_state.messages.append({"role": "user", "content": query}) #message
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in [{"role": "user", "content": message}]#st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.write("RAG Chunks:")
    st.json(context, expanded=False)

    import urllib.parse
    #text_to_copy = response
    #print(text_to_copy)
    #text_to_copy = urllib.parse.quote_plus(text_to_copy)
    #print(text_to_copy)
    #hosted_html_file = "https://neocities.org/site_files/text_editor?filename=copy.html" #"https://everydayswag.org/files/copy.html"
    #iframe_url = f"{hosted_html_file}?copy={text_to_copy}"
#
    #st.markdown(f'<iframe src="{iframe_url}"></iframe>', unsafe_allow_html=True)