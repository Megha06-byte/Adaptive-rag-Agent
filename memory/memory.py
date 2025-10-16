#memory.py
#heap-based PriorityMemory

import heapq
from itertools import count

class PriorityMemory:
    def __init__(self, max_size=10):
        self.heap = []
        self.counter = count()
        self.max_size = max_size

    def add(self, priority, memory):
        heapq.heappush(self.heap, (-priority, next(self.counter), memory))
        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def get_top(self, k=3):
        return [item[2] for item in sorted(self.heap, reverse=True)[:k]]

    def format_memory(self, k=3):
        top_memories = self.get_top(k)
        if not top_memories:
            return ""
        return "\n".join([f"Q: {m['query']}\nA: {m['answer']}" for m in top_memories])
