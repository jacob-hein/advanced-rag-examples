from openai import AzureOpenAI
import streamlit as st

st.title("VtDat Chatbot")

client = AzureOpenAI(api_key=st.secrets["OPENAI_API_KEY"], 
                api_version="2024-05-01-preview", 
                azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"])

###
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from chromadb.config import Settings
from llama_index.core import PromptTemplate

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002",
    api_type="azure",
    api_version="2024-05-01-preview"
)


# Initialize the PersistentClient again with the same path
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

    message = prompt.format(query=query, context="\n\n".join(context))
    print(message)
    ###

    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})