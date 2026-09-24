import http from "http";
import fs from "fs";
import {
  loadModel,
  ragIngest,
  ragSearch,
  completion,
  unloadModel,
  EMBEDDINGGEMMA_300M_Q4_0,
  QWEN3_1_7B_INST_Q4,
} from "@qvac/sdk";

// Silent guardrail: refuse answers when the retrieved context is very weak.
// 0.10 is a safe low value — it only triggers for genuinely unrelated questions.
const MIN_TOP_SCORE = 0.35;

const PORT = 3000;

// ---------- State ----------
let currentWorkspace = null;
let workspaceCounter = 0;
let currentDocumentText = "";

// ---------- HTML ----------
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
    background: #0e0e10; color: #e7e7e9;
    margin: 0; padding: 40px 20px;
    display: flex; justify-content: center;
  }
  .wrap { width: 100%; max-width: 820px; }
  h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: #8a8a92; font-size: 13px; margin-bottom: 24px; }
  .badge {
    display: inline-block; background: #1a1a1d; border: 1px solid #2a2a2e;
    color: #8fd694; font-size: 11px; padding: 3px 8px;
    border-radius: 999px; margin-right: 6px;
  }
  .card {
    background: #17171a; border: 1px solid #26262b;
    border-radius: 12px; padding: 18px; margin-bottom: 16px;
  }
  .label {
    font-size: 12px; color: #8a8a92; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
  }
  textarea, input {
    width: 100%; background: #0e0e10; color: #e7e7e9;
    border: 1px solid #2a2a2e; border-radius: 8px;
    padding: 10px 12px; font-family: inherit; font-size: 14px;
    outline: none; resize: vertical;
  }
  textarea:focus, input:focus { border-color: #4a7cff; }
  #docText {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px; line-height: 1.5; min-height: 200px;
  }
  button {
    background: #4a7cff; color: white; border: 0;
    border-radius: 8px; padding: 10px 18px;
    font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button:disabled { background: #2a2a3e; cursor: not-allowed; }
  .row { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
  .status { font-size: 12px; color: #8a8a92; margin-left: auto; }
  .status.ok { color: #8fd694; }
  .status.err { color: #ff7b7b; }
  #answer { white-space: pre-wrap; line-height: 1.55; font-size: 15px; }
  #answer.refused { color: #ffb86b; font-style: italic; }
  .hidden { display: none; }
  .spinner {
    display: inline-block; width: 12px; height: 12px;
    border: 2px solid #2a2a3e; border-top-color: #4a7cff;
    border-radius: 50%; animation: spin 0.7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty-note {
    color: #6a6a72; font-size: 13px;
    text-align: center; padding: 24px;
    border: 1px dashed #26262b; border-radius: 8px;
    margin-top: 12px;
  }
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
      <div class="label">
        <span>1. Your document</span>
        <span class="status" id="docStatus">Empty</span>
      </div>
      <textarea id="docText" rows="10" placeholder="Paste your document here…"></textarea>
      <div class="row">
        <button id="loadBtn">Load document</button>
        <span class="status" id="loadHint">Paste text above, then click Load.</span>
      </div>
    </div>

    <div class="card">
      <div class="label"><span>2. Ask a question</span></div>
      <textarea id="query" rows="2" placeholder="Type your question about the document…"></textarea>
      <div class="row">
        <button id="askBtn">Ask</button>
        <span class="status" id="askHint">⌘+Enter to send</span>
      </div>
      <div id="notReadyNote" class="empty-note">Load a document first to enable questions.</div>
    </div>

    <div class="card hidden" id="answerCard">
      <div class="label"><span>Answer</span></div>
      <div id="answer"></div>
    </div>
  </div>

<script>
  const docTextEl = document.getElementById("docText");
  const docStatusEl = document.getElementById("docStatus");
  const loadBtn = document.getElementById("loadBtn");
  const loadHintEl = document.getElementById("loadHint");
  const queryEl = document.getElementById("query");
  const askBtn = document.getElementById("askBtn");
  const answerCard = document.getElementById("answerCard");
  const answerEl = document.getElementById("answer");
  const notReadyNote = document.getElementById("notReadyNote");

  let documentLoaded = false;

  function setStatus(el, text, cls) {
    el.textContent = text;
    el.className = "status" + (cls ? " " + cls : "");
  }

  function updateAskState() {
    if (documentLoaded) {
      notReadyNote.classList.add("hidden");
      askBtn.disabled = false;
      queryEl.disabled = false;
    } else {
      notReadyNote.classList.remove("hidden");
      askBtn.disabled = true;
      queryEl.disabled = true;
    }
  }

  // Start with empty document
  docTextEl.value = "";
  updateAskState();

  async function loadDocument() {
    const text = docTextEl.value.trim();
    if (!text) {
      setStatus(docStatusEl, "Cannot load empty document", "err");
      return;
    }
    loadBtn.disabled = true;
    setStatus(docStatusEl, "Ingesting…");
    try {
      const res = await fetch("/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(docStatusEl, "Error: " + data.error, "err");
        documentLoaded = false;
      } else {
        setStatus(docStatusEl, "Ready (" + data.chars + " chars)", "ok");
        setStatus(loadHintEl, "Document loaded. Ask a question below.", "ok");
        documentLoaded = true;
      }
    } catch (e) {
      setStatus(docStatusEl, "Request failed: " + e.message, "err");
      documentLoaded = false;
    } finally {
      loadBtn.disabled = false;
      updateAskState();
    }
  }

  async function ask() {
    const q = queryEl.value.trim();
    if (!q) return;
    askBtn.disabled = true;
    answerCard.classList.remove("hidden");
    answerEl.className = "";
    answerEl.innerHTML = '<span class="spinner"></span>Thinking…';
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
        if (data.refused) {
          answerEl.className = "refused";
          answerEl.textContent = data.answer;
        } else {
          answerEl.className = "";
          answerEl.textContent = data.answer || "(empty)";
        }
      }
    } catch (e) {
      answerEl.textContent = "Request failed: " + e.message;
    } finally {
      askBtn.disabled = false;
    }
  }

  loadBtn.addEventListener("click", loadDocument);
  askBtn.addEventListener("click", ask);
  queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
  });
</script>
</body>
</html>`;

// ---------- Ingest ----------
async function ingestDocument(text, embedModelId) {
  workspaceCounter++;
  const workspace = `privatedoc-${workspaceCounter}`;
  await ragIngest({
    modelId: embedModelId,
    workspace,
    documents: [text],
    chunk: false,
  });
  currentWorkspace = workspace;
  currentDocumentText = text;
}

// ---------- Server ----------
async function main() {
  console.log("Loading embedding model...");
  const embedModelId = await loadModel({
    modelSrc: EMBEDDINGGEMMA_300M_Q4_0,
    modelType: "embeddings",
  });

  console.log("Loading LLM...");
  const llmId = await loadModel({
    modelSrc: QWEN3_1_7B_INST_Q4,
    modelType: "llamacpp-completion",
  });

  const server = http.createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(HTML);
      return;
    }

    if (req.method === "POST" && req.url === "/load") {
      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", async () => {
        try {
          const { text } = JSON.parse(body || "{}");
          if (!text || !text.trim()) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Empty document" }));
            return;
          }
          await ingestDocument(text, embedModelId);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, chars: text.length }));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: String(err.message || err) }));
        }
      });
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
          if (!currentWorkspace) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "No document loaded yet" }));
            return;
          }

          const results = await ragSearch({
            modelId: embedModelId,
            workspace: currentWorkspace,
            query,
            topK: 2,
          });

          const topScore = results.length
            ? Math.max(...results.map((r) => (r && typeof r.score === "number") ? r.score : 0))
            : 0;

          console.log(`[ASK] ${JSON.stringify(query)}  top=${topScore.toFixed(4)}`);

          if (topScore < MIN_TOP_SCORE) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
              refused: true,
              answer: "This isn't covered in the document.",
            }));
            return;
          }

          const context = results
            .map((r) => (typeof r === "string" ? r : r.content || ""))
            .join("\n\n");

          const prompt = `Use the context below to answer the question. Answer in one or two sentences using only information from the context.

Example of good behavior:
Context: "The budget is $45,000 for Q2."
Question: "What is the budget?"
Answer: The budget is $45,000 for Q2.

Example of refusing:
Context: "The budget is $45,000 for Q2."
Question: "What is the CEO's name?"
Answer: This isn't covered in the document.

Now your turn.

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
          answer = answer
            .replace(/<think>[\s\S]*?<\/think>/gi, "")
            .replace(/^Answer:\s*/i, "")
            .trim();

          const refused = /isn'?t covered in the document/i.test(answer);

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ refused, answer }));
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
    console.log("Paste your document in the browser, click Load, then ask.");
    console.log("Press Ctrl+C to stop.");
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
