# Adaptive-rag-Agent
an intelligent adaptive rag agent which can handle queries in an optimal way

# Overview
An intelligent adaptive rag agent which can handle queries in an optimal way. It implements multiple retrieval paths (sparse, dense, hybrid, web) using threading to find the best method for current query, and using hallucination check loop for credible answer to query.

Generated output : 

<img width="1679" height="145" alt="image" src="https://github.com/user-attachments/assets/b773f166-e646-4d4d-85e0-95948f21178b" />
<img width="1704" height="488" alt="image" src="https://github.com/user-attachments/assets/dbeaeb56-3fad-42b9-a1f6-460cd8537c6e" />

# Project structure
├── main.py                  # CLI entrypoint and pipeline orchestrator
├── memory/
│   └── memory.py            # Priority memory management (heap-based)
├── retrieval/
│   └── retrieval.py         # Threaded retrieval (dense, sparse, web)
├── hallucination/
│   └── hallucination.py     # Hallucination scoring and correction
├── data/
│   └── local_data.jsonl     # Local data/cache
├── tests/
│   ├── test_memory.py       # Unit tests for memory
│   ├── test_retrieval.py    # Unit tests for retrieval
│   └── test_hallucination.py# Unit tests for hallucination
├── requirements.txt         # Project dependencies
└── README.md                # Project overview and instructions

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

