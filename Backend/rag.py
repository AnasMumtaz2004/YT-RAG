import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")  # multilingual
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SMALL_VIDEO_CHUNKS = 15  # videos with <= this many chunks: send everything to the LLM

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

prompt = ChatPromptTemplate.from_template(
    """You answer questions about a YouTube video using its transcript excerpts.
The transcript may be auto-generated, so expect spelling mistakes and missing punctuation.
The excerpts may be in a different language than the question.

Rules:
- Use only the excerpts below.
- Reply in the same language as the question.
- If the answer is not in the excerpts, say you don't know.

Transcript excerpts:
{context}

Question: {question}"""
)

chain = prompt | llm | StrOutputParser()

# video_id -> (FAISS index, number of chunks). In memory only: wiped on restart.
stores: dict = {}


def has_video(video_id: str) -> bool:
    return video_id in stores


def ingest(video_id: str, snippets) -> int:
    full_text = " ".join(s.text.replace("\n", " ") for s in snippets)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=c, metadata={"chunk": i}) for i, c in enumerate(chunks)]
    stores[video_id] = (FAISS.from_documents(docs, embeddings), len(docs))
    return len(docs)


def retrieve(video_id: str, question: str, k: int = 4, allow_full: bool = True):
    store, n = stores[video_id]
    if allow_full and n <= SMALL_VIDEO_CHUNKS:
        docs = store.similarity_search(question, k=n)
        return sorted(docs, key=lambda d: d.metadata["chunk"])  # chronological
    return store.similarity_search(question, k=k)


def ask(video_id: str, question: str, k: int = 4, allow_full: bool = True):
    docs = retrieve(video_id, question, k, allow_full)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs