import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

NODE_LABEL     = os.getenv("NODE_LABEL", "Document")
TEXT_PROP      = os.getenv("TEXT_PROP", "text")
EMBED_PROP     = os.getenv("EMBED_PROP", "embedding")
VEC_INDEX_NAME = os.getenv("VEC_INDEX_NAME", "pdf_chunks_vec")
KW_INDEX_NAME  = os.getenv("KW_INDEX_NAME", "pdf_chunks_kw")

OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS)
embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
vector_index = Neo4jVector.from_existing_graph(
    url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS,
    embedding=embed,
    index_name=VEC_INDEX_NAME, keyword_index_name=KW_INDEX_NAME,
    node_label=NODE_LABEL, text_node_properties=[TEXT_PROP],
    embedding_node_property=EMBED_PROP, search_type="hybrid",
)
retriever = vector_index.as_retriever()

class AskRequest(BaseModel):
    question: str = Field(..., description="使用者問題（文字）")

class AskResponse(BaseModel):
    answer: str

class Entities(BaseModel):
    names: List[str] = Field(..., description="All entities from the text")

ent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract organization and person entities from the text. Return JSON."),
    ("human", "Extract entities from: {question}")
])
llm_json = OllamaFunctions(model=OLLAMA_LLM_MODEL, temperature=0)
entity_chain = ent_prompt | llm_json.with_structured_output(Entities, include_raw=False)

def graph_retriever(q: str) -> str:
    try:
        entities = entity_chain.invoke({"question": q}).names
    except Exception:
        entities = []
    if not entities:
        return ""
    lines = []
    for e in entities:
        rows = graph.query("""
            MATCH (p:Person {id: $entity})-[r]->(o)
            RETURN p.id AS s, type(r) AS r, o.id AS t LIMIT 50
        """, {"entity": e})
        lines += [f"{row['s']} - {row['r']} -> {row['t']}" for row in rows]
    return "\n".join(lines)

llm_zh = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是專業的技術助教。無論使用者輸入語言為何，**一律以繁體中文**回覆，"
     "先條列重點，再給簡短總結；若有定義或公式請清楚標示。"),
    ("human", "以下內容：\n{context}\n\n問題：{question}\n請用繁體中文完整回答。")
])

chain = (
    {
        "context": lambda q: "Graph:\n" + graph_retriever(q) + "\n\nVector:\n" +
                             "\n\n".join([f"[Hit {i}] {d.page_content}" for i, d in enumerate(retriever.invoke(q), 1)]),
        "question": RunnablePassthrough(),
    }
    | qa_prompt
    | llm_zh
    | StrOutputParser()
)

app = FastAPI(title="QA Service (Neo4j + Ollama)")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ans = chain.invoke(req.question)
    return AskResponse(answer=ans)