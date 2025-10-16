#retrieval.py
#using multithreading to reduce latency while retreival
#combines sparse, dense and web retrieval methods

import threading
import numpy as np
from rank_bm25 import BM25Okapi
import requests

#BM25 sparse retrieval
def sparse_retrieval(query, chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    best_idx = int(np.argmax(scores))
    return chunks[best_idx], -scores[best_idx]

#web retrieval
def web_retrieval(query):
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("AbstractText"):
            return data["AbstractText"], 0
        elif data.get("RelatedTopics"):
            for topic in data["RelatedTopics"]:
                if isinstance(topic, dict) and topic.get("Text"):
                    return topic["Text"], 1
        return "", float("inf")
    except Exception as e:
        print("Web retrieval error:", e)
        return "", float("inf")
    
#dense retrieval using FAISS vector database
def dense_retrieval(query_embedding, faiss_index, chunks):
    distances, indices = faiss_index.search(np.array([query_embedding]), k=1)
    return chunks[indices[0][0]], distances[0][0]

#shared mutex for results
results = []
lock = threading.lock()

def safe_append(result):
    with lock:
        results.append(result)

#thread creation
dense_thread = threading.Thread(
    target=lambda: safe_append(dense_retrieval(query_embedding, faiss_index, chunks))
)
sparse_thread = threading.Thread(
    target=lambda: safe_append(sparse_retrieval(query, chunks))
)
web_thread = threading.Thread(
    target=lambda: safe_append(web_retrieval(query))
)

#start threads
dense_thread.start()
sparse_thread.start()
web_thread.start()

#waiting for all threads to finish
dense_thread.join()
sparse_thread.join()
web_thread.join()

#best result based on lowest score
best_chunk, best_score = min(results, key=lambda x: x[1])
