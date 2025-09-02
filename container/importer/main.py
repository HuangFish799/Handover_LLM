import os, time, fitz
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

PDF_PATH = os.getenv("PDF_PATH", "/data/paper.pdf")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
GRAPH_CHUNK_CAP = int(os.getenv("GRAPH_CHUNK_CAP", "200"))
GRAPH_BATCH_SIZE = int(os.getenv("GRAPH_BATCH_SIZE", "50"))

INDEX_NAME = os.getenv("VEC_INDEX_NAME", "pdf_chunks_vec")
KW_INDEX_NAME = os.getenv("KW_INDEX_NAME", "pdf_chunks_kw")
NODE_LABEL = os.getenv("NODE_LABEL", "Document")
TEXT_PROP = os.getenv("TEXT_PROP", "text")
EMBED_PROP = os.getenv("EMBED_PROP", "embedding")

def load_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([p.get_text() for p in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def main():
    t0 = time.time()
    print(f"讀取 PDF：{PDF_PATH}")
    chunks = load_pdf_content(PDF_PATH)
    print(f"分段完成，共 {len(chunks)} 段")

    embed = OllamaEmbeddings(model="mxbai-embed-large")
    Neo4jVector.from_texts(
        texts=chunks,
        url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS,
        embedding=embed,
        index_name=INDEX_NAME, keyword_index_name=KW_INDEX_NAME,
        node_label=NODE_LABEL, text_node_property=TEXT_PROP,
        embedding_node_property=EMBED_PROP, search_type="hybrid",
    )
    print("向量索引建立完成")

    limited = chunks[:GRAPH_CHUNK_CAP]
    graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASS)
    transformer = LLMGraphTransformer(llm=ChatOllama(model="llama3", temperature=0))
    for i in range(0, len(limited), GRAPH_BATCH_SIZE):
        batch = [Document(page_content=c) for c in limited[i:i+GRAPH_BATCH_SIZE]]
        graph_docs = transformer.convert_to_graph_documents(batch)
        graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        print(f"已寫入圖譜：{i+len(batch)}/{len(limited)}")

    print(f"匯入完成，總用時 {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()