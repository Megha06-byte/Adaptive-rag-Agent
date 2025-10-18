#hallucination.py
# a corrective loop to check the reliability of generated response, using graphs nodes instead of iterative while loop

from langgraph import StateGraph, Command
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

def generate_answer_agent(state: AgentState):
    client = state.get("client")
    model_name = state.get("model_name")
    query = state.get("query")
    retrieved_chunk = state.get("retrieved_chunk")
    attempt = state.get("attempt", 0)

    # Compose the prompt for the LLM
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.
    Context: {retrieved_chunk}
    Question: {query}
    Answer as accurately and concisely as possible.
    """

    # Call the LLM to generate an answer
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    )
    try:
        new_answer = response.choices[0].message.content.strip()
    except Exception:
        new_answer = f"Failed to generate answer at attempt {attempt}"

    state["answer"] = new_answer
    return Command(goto="hallucination_check_agent", update=state)

# Node: Generate hallucination score
def hallucination_check_agent(state: AgentState):
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
    state["hallucination_score"] = score
    state["attempt"] += 1
    return Command(goto="coordinator_agent", update=state)

# Node: Check if should retry or end
def coordinator_agent(state: AgentState):
    if state["hallucination_score"] > hallucination_threshold and state["attempt"] < max_attempts:
        return Command(goto="generate_answer_agent", update=state)
    else:
        return Command(goto=END, update=state)
        
graph = StateGraph(AgentState)
graph.add_node("generate_answer_agent", generate_answer_agent)
graph.add_node("hallucination_check_agent", hallucination_check_agent)
graph.add_node("coordinator_agent", coordinator_agent)

graph.set_entry_point("generate_answer_agent")

# Initial state example
initial_state = {
    "context": "Some context",
    "answer": "",
    "retrieved_chunk": "Context chunk",
    "hallucination_score": 1.0,
    "attempt": 0,
    "query": "Why is the Amazon rainforest considered important for the global climate, and what are the main threats it faces?",
    "client": client,  # OpenAI client object
    "model_name": "qwen/qwen3-vl-8b-thinking"
}


compiled_graph = graph.compile()
result = compiled_graph.invoke(initial_state)
print(result)
