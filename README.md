# PrivateDoc Q&A

Ask questions about your private documents — fully on-device with Tether's QVAC SDK.

## What it does

Ingests a text document (notes, contract, resume) locally, embeds it on-device, and answers your questions using only that document as context. No server, no API key, no data leaves your machine.

## Requirements

- Node.js >= 22.17.0
- npm >= 10.9
- macOS 14+, Ubuntu 22+, or Windows 10+ (Windows requires Vulkan >= 1.4)
- ~5 GB free disk for model cache

## Install

    npm install

## Run

### Web mode (recommended)

    node src/server.js

Then open http://localhost:3000 in your browser. Paste a document, click Load, then ask questions. AI still runs fully on-device in the Node process.

### Terminal mode

    node src/index.js samples/notes.txt

Then type questions at the prompt. Type exit to quit.

## SDK Version

@qvac/sdk ^0.19.0

## QVAC Functions Used

- loadModel — loads embedding and LLM models
- ragIngest — embeds and stores the document locally
- ragSearch — retrieves relevant chunks for each question
- completion — generates the answer from retrieved context
- unloadModel — frees memory when done

## Offline Demo

After the first run (which downloads the models), turn off Wi-Fi and run again. Everything works because inference is 100% local.

## License

MIT
