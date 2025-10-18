#retrieval.py
#using multithreading to reduce latency while retreival
#combines sparse, dense and web retrieval methods

import threading
import numpy as np
from rank_bm25 import BM25Okapi
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
)

#BM25 sparse retrieval
def sparse_retrieval(query, chunks):
    try:
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        best_idx = int(np.argmax(scores))
        logging.info(f"Sparse retrieval succeeded. Best idx: {best_idx}")
        return chunks[best_idx], -scores[best_idx]
    except Exception as e:
        logging.error(f"Sparse retrieval error: {e}")
        return "", float("inf)

#web retrieval
def web_retrieval(query):
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("AbstractText"):
            logging.info("Web retrieval succeeded with AbstractText.")
            return data["AbstractText"], 0
        elif data.get("RelatedTopics"):
            for topic in data["RelatedTopics"]:
                if isinstance(topic, dict) and topic.get("Text"):
                    logging.info("Web retrieval succeeded with RelatedTopics.")
                    return topic["Text"], 1
        logging.warning("Web retrieval returned no useful result.")
        return "", float("inf")
    except Exception as e:
        logging.error(f"Web retrieval error: {e}")
        return "", float("inf")
    
#dense retrieval using FAISS vector database
def dense_retrieval(query_embedding, faiss_index, chunks):
    try:
        distances, indices = faiss_index.search(np.array([query_embedding]), k=1)
        logging.info(f"Dense retrieval succeeded. Best idx: {indices[0][0]}")
        return chunks[indices[0][0]], distances[0][0]
    except Exception as e:
        logging.error(f"Dense retrieval error: {e}")
        return "", float("inf")
        

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
if results:
    best_chunk, best_score = min(results, key=lambda x: x[1])
    logging.info(f"Best retrieval result selected with score: {best_score}")
else:
    best_chunk, best_score = "", float("inf")
    logging.error("No retrieval results available.")
