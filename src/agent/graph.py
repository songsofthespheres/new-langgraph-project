"""Define the main chatbot agent."""

from typing import Annotated, Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from agent.configuration import Configuration
from agent.state import State
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

def summarizer_agent(state: State) -> Dict:
    """Summarize the legal case."""
    summary = llm.invoke(state["messages"] + [HumanMessage(content="Summarize the legal case presented so far.")])
    return {"messages": state["messages"] + [summary], "current_step": "issue_identification"}

def issue_identifier(state: State) -> Dict:
    """Identify the key legal issues."""
    issues = llm.invoke(state["messages"] + [HumanMessage(content="Identify the key legal issues in this case.")])
    return {"messages": state["messages"] + [issues], "current_step": "research"}

def legal_researcher(state: State) -> Dict:
    """Conduct legal research on the identified issues."""
    research = llm.invoke(state["messages"] + [HumanMessage(content="Conduct legal research on the identified issues.")])
    return {"messages": state["messages"] + [research], "current_step": "analysis"}

def legal_analyzer(state: State) -> Dict:
    """Analyze the case based on the research."""
    analysis = llm.invoke(state["messages"] + [HumanMessage(content="Analyze the case based on the research conducted.")])
    return {"messages": state["messages"] + [analysis], "current_step": "decision"}

def decision_maker(state: State) -> Dict:
    """Make a final decision on the case."""
    decision = llm.invoke(state["messages"] + [HumanMessage(content="Make a final decision on the case based on all the information provided.")])
    return {"messages": state["messages"] + [decision], "current_step": "end"}

def router(state: State) -> str:
    """Route to the next step based on the current_step."""
    return state["current_step"]

# Define the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("issue_identifier", issue_identifier)
workflow.add_node("researcher", legal_researcher)
workflow.add_node("analyzer", legal_analyzer)
workflow.add_node("decision_maker", decision_maker)

# Add edges
workflow.add_edge("summarizer", "issue_identifier")
workflow.add_edge("issue_identifier", "researcher")
workflow.add_edge("researcher", "analyzer")
workflow.add_edge("analyzer", "decision_maker")
workflow.add_edge("decision_maker", END)

# Set conditional edges
workflow.set_entry_point("summarizer")

# Compile the graph
graph = workflow.compile()
