"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypedDict
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage


@dataclass
class State(TypedDict):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    messages: Annotated[List[BaseMessage], "The conversation history"]
    current_step: Annotated[str, "The current step in the legal decision process"]
