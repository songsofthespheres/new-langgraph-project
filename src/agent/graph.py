"""Define the main chatbot agent for Ratio Decidendi Identification."""

from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from agent.state import State
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

def summarizer_agent(state: State) -> Dict:
    """Summarize the legal case."""
    summary = llm.invoke(state["messages"] + [HumanMessage(content="Summarize the key elements of this legal decision, including parties, facts, and outcome.")])
    return {"messages": state["messages"] + [summary], "current_step": "express_issue_identifier"}

def express_issue_identifier(state: State) -> Dict:
    """Identify explicitly stated legal issues."""
    issues = llm.invoke(state["messages"] + [HumanMessage(content="Based on the summary, identify and list all explicitly stated legal issues in this case.")])
    return {"messages": state["messages"] + [issues], "current_step": "decision_extractor"}

def decision_extractor(state: State) -> Dict:
    """Extract the court's decision."""
    decision = llm.invoke(state["messages"] + [HumanMessage(content="Extract and clearly state the court's final decision in this case.")])
    return {"messages": state["messages"] + [decision], "current_step": "argument_identifier"}

def argument_identifier(state: State) -> Dict:
    """Identify key arguments presented by the parties."""
    arguments = llm.invoke(state["messages"] + [HumanMessage(content="Based on the decision and previously identified issues, identify and summarize the key arguments presented by each party in this case.")])
    return {"messages": state["messages"] + [arguments], "current_step": "implicit_issue_identifier"}

def implicit_issue_identifier(state: State) -> Dict:
    """Identify implicit legal issues."""
    implicit_issues = llm.invoke(state["messages"] + [HumanMessage(content="Considering the court's decision, the arguments identified, and the previously identified express issues, identify any implicit legal issues that the court addressed or that are crucial to understanding the decision.")])
    return {"messages": state["messages"] + [implicit_issues], "current_step": "reasoning_tracer"}

def reasoning_tracer(state: State) -> Dict:
    """Trace the court's reasoning process."""
    reasoning = llm.invoke(state["messages"] + [HumanMessage(content="Analyze the court's reasoning process, connecting the identified issues (both express and implicit), arguments, and decision. Highlight key logical steps and legal principles applied.")])
    return {"messages": state["messages"] + [reasoning], "current_step": "initial_ratio_decider"}

def initial_ratio_decider(state: State) -> Dict:
    """Formulate an initial ratio decidendi."""
    initial_ratio = llm.invoke(state["messages"] + [HumanMessage(content="Based on the traced reasoning, formulate an initial ratio decidendi. Focus on the essential rule or principle that was necessary for the court's decision.")])
    return {"messages": state["messages"] + [initial_ratio], "current_step": "material_fact_highlighter"}

def material_fact_highlighter(state: State) -> Dict:
    """Identify material facts based on the initial ratio."""
    material_facts = llm.invoke(state["messages"] + [HumanMessage(content="Given the initial ratio decidendi, identify and highlight the material facts from the case that are crucial to this legal principle.")])
    return {"messages": state["messages"] + [material_facts], "current_step": "final_ratio_decider"}

def final_ratio_decider(state: State) -> Dict:
    """Refine and finalize the ratio decidendi."""
    final_ratio = llm.invoke(state["messages"] + [HumanMessage(content="Considering the initial ratio and the highlighted material facts, refine and finalize the ratio decidendi. Ensure it captures the essential rule of law that was necessary for the court's decision and is grounded in the material facts of the case.")])
    return {"messages": state["messages"] + [final_ratio], "current_step": "end"}

def router(state: State) -> str:
    """Route to the next step based on the current_step."""
    return state["current_step"]

# Define the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("express_issue_identifier", express_issue_identifier)
workflow.add_node("decision_extractor", decision_extractor)
workflow.add_node("argument_identifier", argument_identifier)
workflow.add_node("implicit_issue_identifier", implicit_issue_identifier)
workflow.add_node("reasoning_tracer", reasoning_tracer)
workflow.add_node("initial_ratio_decider", initial_ratio_decider)
workflow.add_node("material_fact_highlighter", material_fact_highlighter)
workflow.add_node("final_ratio_decider", final_ratio_decider)

# Add edges
workflow.add_edge("summarizer", "express_issue_identifier")
workflow.add_edge("express_issue_identifier", "decision_extractor")
workflow.add_edge("decision_extractor", "argument_identifier")
workflow.add_edge("argument_identifier", "implicit_issue_identifier")
workflow.add_edge("implicit_issue_identifier", "reasoning_tracer")
workflow.add_edge("reasoning_tracer", "initial_ratio_decider")
workflow.add_edge("initial_ratio_decider", "material_fact_highlighter")
workflow.add_edge("material_fact_highlighter", "final_ratio_decider")
workflow.add_edge("final_ratio_decider", END)

# Set entry point
workflow.set_entry_point("summarizer")

# Compile the graph
graph = workflow.compile()
