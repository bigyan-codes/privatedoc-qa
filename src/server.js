import http from "http";
import fs from "fs";
import {
  loadModel,
  ragIngest,
  ragSearch,
  completion,
  unloadModel,
  EMBEDDINGGEMMA_300M_Q4_0,
  QWEN3_600M_INST_Q4,
} from "@qvac/sdk";

// ---------- CLI ----------
const args = process.argv.slice(2);
const filePath = args.find((a) => !a.startsWith("--"));
const PORT = 3000;

if (!filePath) {
  console.error("Usage: node src/server.js <document.txt>");
  process.exit(1);
}
if (!fs.existsSync(filePath)) {
  console.error(`ERROR: File not found: ${filePath}`);
  process.exit(1);
}

const documentText = fs.readFileSync(filePath, "utf-8");
if (!documentText.trim()) {
  console.error("ERROR: Document is empty.");
  process.exit(1);
}

// ---------- HTML page ----------
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PrivateDoc Q&amp;A</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #0e0e10;
    color: #e7e7e9;
    margin: 0;
    padding: 40px 20px;
    display: flex;
    justify-content: center;
  }
  .wrap { width: 100%; max-width: 760px; }
  h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: #8a8a92; font-size: 13px; margin-bottom: 24px; }
  .badge {
    display: inline-block;
    background: #1a1a1d;
    border: 1px solid #2a2a2e;
    color: #8fd694;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 999px;
    margin-right: 6px;
  }
  .card {
    background: #17171a;
    border: 1px solid #26262b;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
  }
  .label { font-size: 12px; color: #8a8a92; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
  textarea, input {
    width: 100%;
    background: #0e0e10;
    color: #e7e7e9;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 10px 12px;
    font-family: inherit;
    font-size: 14px;
    outline: none;
    resize: vertical;
  }
  textarea:focus, input:focus { border-color: #4a7cff; }
  button {
    background: #4a7cff;
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 12px;
  }
  button:disabled { background: #2a2a3e; cursor: not-allowed; }
  .doc-preview {
    background: #0e0e10;
    border: 1px solid #222226;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: #9a9aa2;
    white-space: pre-wrap;
    max-height: 160px;
    overflow-y: auto;
    line-height: 1.5;
  }
  #answer {
    white-space: pre-wrap;
    line-height: 1.55;
    font-size: 15px;
  }
  #sources {
    font-size: 12px;
    color: #8a8a92;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #222226;
  }
  .hidden { display: none; }
  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #2a2a3e; border-top-color: #4a7cff; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="wrap">
    <h1>PrivateDoc Q&amp;A</h1>
    <div class="sub">
      <span class="badge">100% on-device</span>
      <span class="badge">QVAC SDK</span>
      <span class="badge">no network</span>
    </div>

    <div class="card">
      <div class="label">Your document</div>
      <div class="doc-preview" id="docPreview">Loading…</div>
    </div>

    <div class="card">
      <div class="label">Ask a question</div>
      <textarea id="query" rows="2" placeholder="What is the budget?"></textarea>
      <button id="askBtn">Ask</button>
    </div>

    <div class="card hidden" id="answerCard">
      <div class="label">Answer</div>
      <div id="answer"></div>
      <div id="sources"></div>
    </div>
  </div>

<script>
  const queryEl = document.getElementById("query");
  const askBtn = document.getElementById("askBtn");
  const answerCard = document.getElementById("answerCard");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");
  const docPreviewEl = document.getElementById("docPreview");

  fetch("/document").then(r => r.text()).then(t => {
    docPreviewEl.textContent = t;
  });

  async function ask() {
    const q = queryEl.value.trim();
    if (!q) return;
    askBtn.disabled = true;
    answerCard.classList.remove("hidden");
    answerEl.innerHTML = '<span class="spinner"></span>Thinking…';
    sourcesEl.textContent = "";
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();
      if (data.error) {
        answerEl.textContent = "Error: " + data.error;
      } else {
        answerEl.textContent = data.answer || "(empty)";
        sourcesEl.textContent = "Retrieved " + data.resultCount + " chunk(s) from your document.";
      }
    } catch (e) {
      answerEl.textContent = "Request failed: " + e.message;
    } finally {
      askBtn.disabled = false;
    }
  }

  askBtn.addEventListener("click", ask);
  queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
  });
</script>
</body>
</html>`;

// ---------- HTTP server ----------
async function main() {
  console.log("Loading embedding model...");
  const embedModelId = await loadModel({
    modelSrc: EMBEDDINGGEMMA_300M_Q4_0,
    modelType: "embeddings",
  });

  console.log("Loading LLM...");
  const llmId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelType: "llamacpp-completion",
  });

  console.log("Ingesting document...");
  const workspace = "privatedoc";
  await ragIngest({
    modelId: embedModelId,
    workspace,
    documents: [documentText],
    chunk: false,
  });

  const server = http.createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(HTML);
      return;
    }

    if (req.method === "GET" && req.url === "/document") {
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(documentText);
      return;
    }

    if (req.method === "POST" && req.url === "/ask") {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", async () => {
        try {
          const { query } = JSON.parse(body || "{}");
          if (!query || !query.trim()) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Empty query" }));
            return;
          }

          const results = await ragSearch({
            modelId: embedModelId,
            workspace,
            query,
            topK: 3,
          });

          const context = results
            .map((r) => (typeof r === "string" ? r : r.content || ""))
            .join("\n\n");

          const prompt = `Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
${context}

Question: ${query}

Answer:`;

          const stream = completion({
            modelId: llmId,
            history: [{ role: "user", content: prompt }],
            stream: true,
            temperature: 0,
          });

          let answer = "";
          for await (const token of stream.tokenStream) answer += token;
          answer = answer.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/^Answer:\s*/i, "").trim();

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ answer, resultCount: results.length }));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: String(err.message || err) }));
        }
      });
      return;
    }

    res.writeHead(404);
    res.end("Not found");
  });

  server.listen(PORT, () => {
    console.log(`\n✅ Ready → http://localhost:${PORT}\n`);
    console.log("Open that URL in your browser. Press Ctrl+C to stop.");
  });

  let shuttingDown = false;
  process.on("SIGINT", async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log("\nShutting down...");
    server.close();
    try { await unloadModel({ modelId: llmId }); } catch (e) {}
    try { await unloadModel({ modelId: embedModelId }); } catch (e) {}
    process.exit(0);
  });
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
