"""Define the main chatbot agent for Ratio Decidendi Identification."""

from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from agent.state import State
from langchain_openai import ChatOpenAI
import json

# Initialize the LLM with a lower temperature for more focused outputs
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def summarizer_agent(state: State) -> Dict:
    """Summarize the key elements of this legal decision."""
    summary = llm.invoke(state["messages"] + [HumanMessage(content="""
    Provide a concise summary of the key elements of the following legal judgment. The summary should be about 150 words long. Provide output in JSON format as follows:
    {"summary": "..."}

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=summary.content)], "current_step": "express_issue_identifier"}

def express_issue_identifier(state: State) -> Dict:
    """Identify explicitly stated legal issues."""
    issues = llm.invoke(state["messages"] + [HumanMessage(content="""
    Identify and extract all relevant direct quotations from the following legal judgment that explicitly state the legal issues addressed in the case. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"issue": "..."}, ..., {"issue": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=issues.content)], "current_step": "decision_extractor"}

def decision_extractor(state: State) -> Dict:
    """Extract the court's decision."""
    decision = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations from the following legal judgment that detail the court's final decision. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"decision": "..."}, ..., {"decision": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=decision.content)], "current_step": "argument_identifier"}

def argument_identifier(state: State) -> Dict:
    """Identify key arguments from each party."""
    arguments = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations from the following legal judgment that present the key arguments from each party involved in the case. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"party": "...", "argument": "..."}, ..., {"party": "...", "argument": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=arguments.content)], "current_step": "implicit_issue_identifier"}

def implicit_issue_identifier(state: State) -> Dict:
    """Identify implicit legal issues."""
    issues = llm.invoke(state["messages"] + [HumanMessage(content="""
    Identify and extract any implicit legal issues from the following legal judgment. You may include minimal commentary to explain their relevance. Provide output in JSON format as follows:
    [{"issue": "...", "explanation": "..."}, ..., {"issue": "...", "explanation": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=issues.content)], "current_step": "reasoning_tracer"}

def reasoning_tracer(state: State) -> Dict:
    """Trace the court's reasoning process."""
    reasoning = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations that demonstrate the court's reasoning process. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"step": "...", "reasoning": "..."}, ..., {"step": "...", "reasoning": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=reasoning.content)], "current_step": "initial_ratio_decider"}

def initial_ratio_decider(state: State) -> Dict:
    """Determine the initial ratio decidendi."""
    ratio = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations that compose the initial ratio decidendi of the case. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"ratio": "..."}, ..., {"ratio": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=ratio.content)], "current_step": "material_fact_highlighter"}

def material_fact_highlighter(state: State) -> Dict:
    """Highlight material facts of the case."""
    facts = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations that highlight the material facts of the case. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"fact": "..."}, ..., {"fact": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=facts.content)], "current_step": "final_ratio_decider"}

def final_ratio_decider(state: State) -> Dict:
    """Determine the final ratio decidendi."""
    final_ratio = llm.invoke(state["messages"] + [HumanMessage(content="""
    Extract all relevant direct quotations that compose the final ratio decidendi of the case, considering the initial ratio and the highlighted material facts. Do not include any paraphrasing or additional commentary. Provide output in JSON format as follows:
    [{"ratio": "..."}, ..., {"ratio": "..."}]

    Judgment Text:
    {text}
    """)])
    return {"messages": state["messages"] + [AIMessage(content=final_ratio.content)], "current_step": "end"}

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
