### 1. 在專案根目錄執行

```bash
docker compose up --build
```

- 流程會自動：
    - 啟動 Neo4j、Ollama
    - ollama-init 下載 `llama3` 與 `mxbai-embed-large`
    - importer 讀取 `data/paper.pdf` → 建立 `:Document(text, embedding)`、向量/全文索引、知識圖，完成後結束
    - 啟動 QA 服務，提供 API

### 2. 檢查服務狀態

```bash
GET http://localhost:8000/healthz
→ {"ok": true}
```

### 3. 問答

```bash
POST http://localhost:8000/ask
Content-Type: application/json

{"question": "RUL 是什麼？"}

{"answer": "（繁體中文條列＋總結的回答）"}
```

### 4. 重啟服務（已匯入過 data）

```bash
docker compose up -d neo4j ollama qa
```