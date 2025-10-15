# Adaptive-rag-Agent
# Overview
An intelligent adaptive rag agent which can handle queries in an optimal way. It implements multiple retrieval paths (sparse, dense, hybrid, web) using threading to find the best method for current query, and using hallucination check loop for credible answer to query.

Input :

Text :
The Amazon rainforest, often referred to as the 'lungs of the Earth,' is the largest tropical rainforest in the world, spanning over nine countries in South America and covering approximately 5.5 million square kilometers. It is home to an incredibly diverse array of plant and animal species, many of which are found nowhere else on the planet. The rainforest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide and producing oxygen. However, the Amazon faces significant threats from deforestation, illegal logging, mining, and agricultural expansion. These activities not only endanger countless species but also disrupt the livelihoods of indigenous communities who have lived in harmony with the forest for generations. Conservation efforts are underway, but the challenge remains immense due to economic pressures and political complexities in the region.

Query : 
Why is the Amazon rainforest considered important for the global climate, and what are the main threats it faces?

Generated output : 

<img width="1679" height="145" alt="image" src="https://github.com/user-attachments/assets/b773f166-e646-4d4d-85e0-95948f21178b" />
<img width="1704" height="488" alt="image" src="https://github.com/user-attachments/assets/dbeaeb56-3fad-42b9-a1f6-460cd8537c6e" />

# Project structure
```├── main.py                  # CLI entrypoint and pipeline orchestrator
├── memory/
│   └── memory.py            # Priority memory management (heap-based)
├── retrieval/
│   └── retrieval.py         # Threaded retrieval (dense, sparse, web)
├── hallucination/
│   └── hallucination.py     # Hallucination scoring and correction
├── requirements.txt         # Project dependencies
└── README.md                # Project overview and instructions
```


# Pipeline
                  +-------------------+
                  |    User Query     |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  |  Threaded Retrieval|
                  | (Dense/Sparse/Web)|
                  +-------------------+
                           |
                           v
                  +-------------------+
                  |  Priority Memory  |
                  |   (Heap Buffer)   |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  |   LLM Generation  |
                  +-------------------+
                           |
                           v
                  +------------------------+
                  | Hallucination Check    |
                  | (Loop if needed)       |
                  +------------------------+
                           |
                           v
                  +-------------------+
                  |   Final Answer    |
                  +-------------------+

# Tech stack

```Component              |  Technology/Tool                          |  Model/Library Used         |  Purpose/Description              
-----------------------+-------------------------------------------+-----------------------------+-----------------------------------
Programming Language   |  Python                                   |  —                          |  Scripting, orchestration         
Orchestration/Agents   |  LangChain, LangGraph (optional)          |  —                          |  Workflow control, agentic logic  
Retrieval              |  FAISS (Dense), BM25 (Sparse), Threading  |  numpy, rank_bm25           |  Hybrid and parallel retrieval    
Memory Management      |  Custom Heap, Priority Buffer             |  heapq, itertools           |  Selects/stores top-k memories    
LLM Interface          |  OpenRouter Python SDK                    |  qwen/qwen3-vl-8b-thinking  |  Runs LLM for answering, scoring  
Embedding Interface    |  Hugging Face                             |  all-MiniLM-L6-v2           |  Sentence embeddings for retrieval
Hallucination Scoring  |  Custom LLM prompt (OpenRouter)           |  qwen/qwen3-vl-8b-thinking  |  Checks/corrects LLM generations  
Chunking/Splitting     |  LangChain TextSplitter                   |  langchain.text_splitter    |  Context preprocessing
```    
