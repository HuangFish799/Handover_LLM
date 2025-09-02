import os
from typing import List

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "YourPassword")

# 需與 import.py 中的設定一致
NODE_LABEL = "Document"
TEXT_PROP = "text"
EMBED_PROP = "embedding"
VEC_INDEX_NAME = "pdf_chunks_vec"
KW_INDEX_NAME = "pdf_chunks_kw"

OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"

def init_graph_and_retriever():
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS)

    embed = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    try:
        vector_index = Neo4jVector.from_existing_graph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASS,
            embedding=embed,
            index_name=VEC_INDEX_NAME,
            keyword_index_name=KW_INDEX_NAME,
            node_label=NODE_LABEL,
            text_node_properties=[TEXT_PROP],
            embedding_node_property=EMBED_PROP,
            search_type="hybrid",
        )
    except Exception as e:
        raise RuntimeError(
            f"[初始化失敗] 請確認 Neo4j 內已存在向量索引 `{VEC_INDEX_NAME}` 與關鍵字索引 `{KW_INDEX_NAME}`，"
            f"且都指向 :{NODE_LABEL}({TEXT_PROP})／向量屬性 {EMBED_PROP}。原始錯誤：{e}"
        )

    retriever = vector_index.as_retriever()
    return graph, retriever

class Entities(BaseModel):
    names: List[str] = Field(..., description="All entities from the text")

def build_qa_chain(graph, retriever):
    # 實體抽取，僅用於圖查詢，不參與回答語境
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
                lines.append(f"{el['source_id']} - {el['relationship']} -> {el['target_id']}")
        return "\n".join(lines)

    # 最終回答（繁體中文）
    llm_zh = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)

    def full_retriever(q: str) -> str:
        g = graph_retriever(q)
        try:
            hits = retriever.invoke(q)
        except Exception:
            hits = []
        vec_texts = [f"[Chunk {i}] {doc.page_content}" for i, doc in enumerate(hits, 1)]
        return f"Graph data:\n{g}\n\nVector data:\n" + "\n\n".join(vec_texts)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是專業的技術助教。無論使用者輸入語言為何，**一律以繁體中文**回覆，"
         "請：1) 先條列重點、2) 再給一段簡短總結、3) 若有定義或公式，請清楚標示。"),
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
    return chain

def main():
    print("連線到既有 Neo4j 與向量索引（不重建）...")
    graph, retriever = init_graph_and_retriever()
    chain = build_qa_chain(graph, retriever)
    print("輸入你的問題（空白直接離開）\n")

    try:
        while True:
            q = input("問題> ").strip()
            if not q:
                break
            ans = chain.invoke(q)
            print("\n================ 回答（繁中） ================\n")
            print(ans)
            print("\n=============================================\n")
    except KeyboardInterrupt:
        pass
    finally:
        print("結束")


if __name__ == "__main__":
    main()