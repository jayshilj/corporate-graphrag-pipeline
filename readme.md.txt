# 🚀 Corporate GraphRAG Strategy Analyzer

A specialized **Knowledge Graph Retrieval-Augmented Generation (GraphRAG)** pipeline designed to perform multi-hop reasoning across corporate history, intellectual property (IP), and R&D data.

## 🧠 Why GraphRAG?
Standard Vector-based RAG often fails at complex, relational questions. For example, if asked *"How does a founder's 2010 research influence a 2026 patent?"*, a standard RAG might find the paper or the patent, but it struggles to bridge the connection.

This project solves that by building a **Knowledge Graph**, physically mapping relationships between entities to allow the LLM to "walk the path" from a person to a concept, and finally to a corporate asset.



## 🛠️ The Tech Stack
* **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
* **LLM:** [Groq](https://groq.com/) (Llama 3.3 70B - for high-speed, high-accuracy entity extraction)
* **Graph Database:** [Neo4j AuraDB](https://neo4j.com/cloud/aura/) (Cloud-native graph storage)
* **Embeddings:** [HuggingFace](https://huggingface.co/) (Local `bge-small-en-v1.5` - cost-effective & privacy-focused)

## 🏗️ System Architecture
1.  **Ingestion:** Raw text documents (SEC filings, R&D papers) are parsed.
2.  **Extraction:** Llama 3.3 extracts entities (`PERSON`, `COMPANY`, `PATENT`) based on a strict defined schema.
3.  **Graph Construction:** Entities and their relationships (`FOUNDED`, `AUTHORED`, `OWNS_PATENT`) are stored in Neo4j.
4.  **Querying:** The system traverses the graph to retrieve contextually linked data before generating a response.

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* A free Neo4j AuraDB instance.
* A free Groq API key.

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
NEO4J_URI=bolt://your-instance-url:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
GROQ_API_KEY=gsk_your_key_here

![Graph Visualization](./images/graph_viz.png)