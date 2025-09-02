import os
import time
import fitz
from typing import List

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "YourPassword")

OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"

PDF_PATH = "YourPdfFileName.pdf"

# 切分大小
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# 圖譜抽取限額與批次
GRAPH_CHUNK_CAP = 200
GRAPH_BATCH_SIZE = 50

# 初始化 Neo4j Graph
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS
)
vector_retriever = None

def _page_text_or_ocr(page) -> str:
    text = page.get_text()
    if text.strip():
        return text
    if not ENABLE_OCR_FALLBACK:
        return ""

def load_pdf_content(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(_page_text_or_ocr(page))
    full_text = "\n".join(texts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(full_text)
    return chunks

def ingestion_from_pdf(pdf_path: str,
                       graph_chunk_cap: int = GRAPH_CHUNK_CAP,
                       batch_size: int = GRAPH_BATCH_SIZE):
    start_time = time.time()
    chunks = load_pdf_content(pdf_path)
    print(f"分段完成，共 {len(chunks)} 段")

    # 先建立向量索引
    print("建立向量索引...")
    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    # 直接將文字寫入 Neo4j 的向量索引
    # 注意：不同版的 Neo4jVector 可能參數略有差異；以下為常見寫法
    vector_index = Neo4jVector.from_texts(
        texts=chunks,
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASS,
        embedding=embed,
        index_name="pdf_chunks_vec",
        keyword_index_name="pdf_chunks_kw",
        node_label="Document",
        text_node_property="text",
        embedding_node_property="embedding",
        search_type="hybrid",
    )
    global vector_retriever
    vector_retriever = vector_index.as_retriever()
    print("向量索引建立完成")

    # 圖譜抽取，並分批寫入
    limited = chunks[:graph_chunk_cap]
    print(f"開始將前 {len(limited)} 段轉為圖文件（每批 {batch_size}）...")
    llm_for_graph = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)
    transformer = LLMGraphTransformer(llm=llm_for_graph)

    for i in range(0, len(limited), batch_size):
        batch = [Document(page_content=c) for c in limited[i:i+batch_size]]
        graph_docs = transformer.convert_to_graph_documents(batch)
        graph.add_graph_documents(
            graph_docs, baseEntityLabel=True, include_source=True)
        print(f"已寫入圖譜：{i+len(batch)}/{len(limited)}")

    print(f"總處理時間：{time.time() - start_time:.2f} 秒")

def querying_neo4j(question: str):
    # 實體抽取，僅用於檢索，不參與回答語境
    class Entities(BaseModel):
        names: list[str] = Field(..., description="All entities from the text")

    ent_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract organization and person entities from the text. Return JSON."),
        ("human", "Extract entities from: {question}")
    ])
    llm_json = OllamaFunctions(model=OLLAMA_LLM_MODEL, temperature=0)
    entity_chain = ent_prompt | llm_json.with_structured_output(
        Entities, include_raw=False)

    def graph_retriever(q: str) -> str:
        try:
            entities = entity_chain.invoke({"question": q}).names
        except Exception:
            entities = []
        if not entities:
            return ""
        print("[實體抽取] ", entities)
        result_lines = []
        for entity in entities:
            rows = graph.query(
                """
                MATCH (p:Person {id: $entity})-[r]->(e)
                RETURN p.id AS source_id, type(r) AS relationship, e.id AS target_id
                LIMIT 50
                """,
                {"entity": entity}
            )
            for el in rows:
                result_lines.append(
                    f"{el['source_id']} - {el['relationship']} -> {el['target_id']}")
        return "\n".join(result_lines)

    # 最終回答（繁體中文）
    llm_zh = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)

    def full_retriever(q: str) -> str:
        graph_data = graph_retriever(q)
        try:
            vector_hits = vector_retriever.invoke(q)
        except Exception:
            vector_hits = []
        vector_data = []
        for idx, doc in enumerate(vector_hits, 1):
            vector_data.append(f"[Chunk {idx}] {doc.page_content}")
        return f"Graph data:\n{graph_data}\n\nVector data:\n" + "\n\n".join(vector_data)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是專業的技術助教。無論使用者輸入語言為何，**一律以繁體中文**回覆，"
         "用條列清楚重點，最後給一段簡短總結。"),
        ("human", "以下是檢索到的內容：\n{context}\n\n問題：{question}\n請用繁體中文完整作答。")
    ])

    chain = (
        {
            "context": lambda input: full_retriever(input),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm_zh
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    print("\n================ 最終回答 ================\n")
    print(answer)
    print("\n==========================================\n")

if __name__ == "__main__":
    # 可視需要清除既有資訊
    # print("清除舊有資料中...")
    # graph.query("MATCH (n) DETACH DELETE n")

    print("載入 PDF 並建立圖與向量...")
    ingestion_from_pdf(PDF_PATH, graph_chunk_cap=GRAPH_CHUNK_CAP,
                       batch_size=GRAPH_BATCH_SIZE)
    print("資料匯入完成！\n")

    print("啟動問答...")
    querying_neo4j("RUL是什麼？")
    print("問答完成。")