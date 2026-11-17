#state.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


#==================================
# 2. Define the Graph State
#==================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    grader_retries: int