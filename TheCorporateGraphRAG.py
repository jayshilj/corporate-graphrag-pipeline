import os
from typing import Literal
from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# 1. Load environment variables
load_dotenv()

# 2. Setup Groq LLM and Local Embeddings
# Using Groq's current active model: Llama 3.3 70B
llm = Groq(model="llama-3.3-70b-versatile", temperature=0)

# Local embeddings (Already downloaded and cached on your machine!)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# 3. Connect to Neo4j Graph Database
# database=None forces it to use your active cloud instance
graph_store = Neo4jPropertyGraphStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    url=os.getenv("NEO4J_URI"),
    database=None, 
)

# 4. Define the Strict Schema using Python Literals
entities = Literal["PERSON", "COMPANY", "RESEARCH_PAPER", "PATENT", "TECHNOLOGY_CONCEPT"]
relations = Literal["FOUNDED", "AUTHORED", "OWNS_PATENT", "CITES_CONCEPT", "FOCUSES_ON"]

validation_schema = {
    "PERSON": ["FOUNDED", "AUTHORED"],
    "COMPANY": ["OWNS_PATENT", "FOCUSES_ON"],
    "RESEARCH_PAPER": ["FOCUSES_ON"],
    "PATENT": ["CITES_CONCEPT", "FOCUSES_ON"],
    "TECHNOLOGY_CONCEPT": []
}

# 5. Initialize the Extractor
schema_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
    num_workers=1 
)

# 6. Dummy Financial & IP Documents
docs = [
    Document(text="In 2010, Dr. Elena Rostova published 'Quantum Edge Routing Algorithms' focusing on low-latency data transmission."),
    Document(text="Dr. Elena Rostova founded NexaCorp in 2015 to commercialize quantum routing technologies."),
    Document(text="NexaCorp's 2026 Q1 SEC Filing: The company is aggressively expanding its IP portfolio. NexaCorp recently filed Patent US-2026-998, which covers 'Predictive Quantum Node Hopping'."),
    Document(text="Patent US-2026-998 explicitly builds upon foundational low-latency data transmission concepts to reduce server load.")
]

def build_financial_graph():
    print("Extracting IP & Corporate entities into Neo4j using Groq (Llama 3.3)...")
    index = PropertyGraphIndex.from_documents(
        docs,
        kg_extractors=[schema_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )
    return index

def query_financial_strategy(index, question):
    print(f"\nAnalyzing Query: {question}")
    
    # FIX APPLIED: Pass settings directly to avoid the 'retriever' argument clash
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=3,
        llm=llm
    )
    
    response = query_engine.query(question)
    print("\n--- Strategy Analysis ---")
    print(response.response)

if __name__ == "__main__":
    # Since you successfully ran the extraction in the last step, 
    # we leave the build step commented out so we don't duplicate nodes.
    # index = build_financial_graph()
    
    # Load your freshly built knowledge graph directly from Neo4j
    print("Connecting to Neo4j to load existing Graph...")
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
    )
    
    # Ask the multi-hop question
    query_financial_strategy(
        index, 
        "How does the founder's early research in 2010 affect the company's current 2026 patent strategy?"
    )