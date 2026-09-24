import {
  loadModel,
  ragIngest,
  ragSearch,
  completion,
  unloadModel,
  EMBEDDINGGEMMA_300M_Q4_0,
  QWEN3_600M_INST_Q4,
} from "@qvac/sdk";
import fs from "fs";
import readline from "readline";

// ---------- CLI arguments ----------
const args = process.argv.slice(2);
const filePath = args.find((a) => !a.startsWith("--"));
const keepModels = args.includes("--keep-models");

if (!filePath || args.includes("--help")) {
  console.log(`
Usage: node src/index.js <document.txt> [options]

Options:
  --keep-models     skip unloadModel (debugging)
  --help            show usage

Example:
  node src/index.js samples/notes.txt
  Then type your questions at the prompt.
  `);
  process.exit(filePath ? 0 : 1);
}

if (!fs.existsSync(filePath)) {
  console.error(`ERROR: File not found: ${filePath}`);
  process.exit(1);
}

// ---------- Progress helper ----------
function onProgress(label) {
  return (p) => {
    if (p && typeof p === "object" && p.progress != null) {
      process.stdout.write(`\r${label}: ${Math.round(p.progress * 100)}%`);
    }
  };
}

// ---------- Main pipeline ----------
async function main() {
  console.log("loading models -> ingesting document -> ready for questions\n");

  const documentText = fs.readFileSync(filePath, "utf-8");

  if (!documentText.trim()) {
    console.error("ERROR: Document is empty.");
    process.exit(1);
  }

  // 1. Load embedding model
  console.log("Loading embedding model...");
  const embedModelId = await loadModel({
    modelSrc: EMBEDDINGGEMMA_300M_Q4_0,
    modelType: "embeddings",
    onProgress: onProgress("Embeddings"),
  });

  // 2. Load LLM for answering
  console.log("\nLoading LLM...");
  const llmId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelType: "llamacpp-completion",
    onProgress: onProgress("LLM"),
  });

  // 3. Ingest the document into a RAG workspace
  console.log("\nIngesting document...");
  const workspace = "privatedoc";
  await ragIngest({
    modelId: embedModelId,
    workspace: workspace,
    documents: [documentText],
    chunk: false,
  });
  console.log("Document ingested. You can now ask questions.\n");

  // 4. Interactive Q&A loop
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const askQuestion = () => {
    rl.question("Ask a question (or type 'exit' to quit): ", async (query) => {
      if (query.trim().toLowerCase() === "exit") {
        rl.close();
        if (!keepModels) {
          await unloadModel({ modelId: embedModelId });
          await unloadModel({ modelId: llmId });
          console.log("Models unloaded. Done.");
        }
        return;
      }

      if (!query.trim()) {
        askQuestion();
        return;
      }

      // Retrieve relevant chunks from the document
      const results = await ragSearch({
        modelId: embedModelId,
        workspace: workspace,
        query: query,
        topK: 3,
      });

      const context = results
        .map((r) => (typeof r === "string" ? r : r.content || ""))
        .join("\n\n");

      // Ask the LLM to answer using only the retrieved context
      const prompt = `Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
${context}

Question: ${query}

Answer:`;

      const result = completion({
        modelId: llmId,
        history: [{ role: "user", content: prompt }],
        stream: true,
        temperature: 0,
      });

      let answerText = "";
      for await (const token of result.tokenStream) {
        answerText += token;
      }

      // Strip Qwen3's internal <think>...</think> reasoning block
      answerText = answerText.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/^Answer:\s*/i, "").trim();

      console.log("\n--- ANSWER ---");
      console.log(answerText || "(empty)");
      console.log("--- END ---\n");

      askQuestion();
    });
  };

  askQuestion();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
