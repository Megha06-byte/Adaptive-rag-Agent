#hallucination.py
# a corrective loop to check the reliability of generated response, using graphs nodes instead of iterative while loop

from langgraph import StateGraph
from langgraph.types import END, TypedDict
from openai import OpenAI

#state schema
class AgentState(TypedDict):
    context: str
    answer: str
    retrieved_chunk: str
    hallucination_score: float
    attempt: int

client = OpenAI(api_key="API_KEY", base_url="https://openrouter.ai/api/v1")
model_name = "qwen/qwen3-vl-8b-thinking"
max_attempts = 5
hallucination_threshold = 0.2

# Node: Generate hallucination score
def hallucination_check(state: AgentState):
    hallucination_prompt = f"""
    You are an expert assistant helping to check if statements are based on the context.
    Context: {state['retrieved_chunk']}
    Statement: {state['answer']}

    Provide a hallucination score between 0 and 1, where 0 means fully grounded in context and 1 means fully hallucinated.
    Only provide the score.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [{"type": "text", "text": hallucination_prompt}]}]
    )
    try:
        score = float(response.choices[0].message.content.strip())
    except Exception:
        score = 1.0
    return {"hallucination_score": score, "attempt": state["attempt"] + 1}

# Node: Check if should retry or end
def decide_next(state: AgentState):
    if state["hallucination_score"] > hallucination_threshold and state["attempt"] < max_attempts:
        return "Regenerate"
    else:
        return END
    
# Node: Regenerate answer (stub here, implement your own generation)
def regenerate(state: AgentState):
    # Example regeneration logic (replace with your LLM generation call)
    new_answer = f"Regenerated answer at attempt {state['attempt']}"
    return {"answer": new_answer}
    
# Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("HallucinationCheck", hallucination_check)
graph.add_node("DecideNext", decide_next)
graph.add_node("Regenerate", regenerate)

graph.add_edge("HallucinationCheck", "DecideNext", condition=True)
graph.add_conditional_edges("DecideNext", lambda state: state, {"Regenerate": "Regenerate", END: END})
graph.add_edge("Regenerate", "HallucinationCheck")
graph.set_entry_point("HallucinationCheck")

#initial state example
initial_state = {
    "context": "Some context",
    "answer": "Initial LLM answer",
    "retrieved_chunk": "Context chunk",
    "hallucination_score": 1.0,
    "attempt": 0,
}

compiled_graph = graph.compile()
result = compiled_graph.invoke(initial_state)
print(result)
