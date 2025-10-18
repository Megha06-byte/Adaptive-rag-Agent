from memory import PriorityMemory
from retrieval import threaded_retrieval
from hallucination import (
    generate_answer_agent,
    hallucination_check_agent,
    coordinator_agent,
    graph as hallucination_graph,
    AgentState,
    END,
)

from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
from openai import OpenAI

# Initialize embedding model and split context
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
paragraph = """The Amazon rainforest, often referred to as the 'lungs of the Earth,' ..."""  # Your large context here

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_text(paragraph)
embeddings = embedding_model.encode(chunks)
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(np.array(embeddings))

# Initialize agent state and memory buffer
agent_state = {
    "context": "",  # Optional
    "answer": "",
    "retrieved_chunk": retrieved_chunk,  # Set after retrieval
    "hallucination_score": 1.0,
    "attempt": 0,
    "query": query,
    "client": client,
    "model_name": "qwen/qwen3-vl-8b-thinking"
}

query = "Why is the Amazon rainforest considered important for the global climate, and what are the main threats it faces?"
agent_state["query"] = query  # Add query 

# Embed the query
query_embedding = embedding_model.encode([query])[0]

# Perform threaded retrieval to get best context chunk
retrieved_chunk = threaded_retrieval(query_embedding, faiss_index, chunks, query)
agent_state["retrieved_chunk"] = retrieved_chunk

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="API_KEY",
)

# Add client and model name to agent state for use in nodes if needed
agent_state["client"] = client  
agent_state["model_name"] = "qwen/qwen3-vl-8b-thinking"
agent_state["memory_buffer"] = PriorityMemory(max_size=5)

# Compile LangGraph with custom state
compiled_graph = hallucination_graph.compile()

# Invoke the graph, which encapsulates regeneration loop
result_state = compiled_graph.invoke(agent_state)

# Extract final answer and hallucination score
final_answer = result_state.get("answer", "")
final_score = result_state.get("hallucination_score", 1.0)

print(f"Final Hallucination Score: {final_score}")
if final_score <= 0.2:
    print("Final answer accepted (low hallucination):", final_answer)
else:
    print("Failed to generate sufficiently grounded answer after max attempts.")
