import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

import io
import contextlib

# LangGraph & Core Components
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# Custom Files Imports
from custom_tools import legal_agent
from prompt import SYSTEM_MESSAGE
from state import State
from grader import grader_node

load_dotenv()

#==================================
# 1. LLM Setup
#==================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

all_rag_tools = [legal_agent]
llm_with_tools = llm.bind_tools(all_rag_tools)  #agent


#==================================
# 3. Nodes
#==================================
def assistant(state: State):
    """The 'Brain': Calls tools or generates final response."""
    # Use the original SYSTEM_MESSAGE from your prompt.py
    response = llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}



#user passes the question
#possibilty 1- tool is called.- tool has to be called .


#===================================
# Routing Logic
#===================================
def route_assistant(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. If tool call requested, go to tools
    if last_message.tool_calls:
        return "tools"
    
    # 2. If the last message was a "No" from the grader, force back to assistant
    if isinstance(last_message, AIMessage) and "no" in last_message.content.lower():
        return "assistant"
        
    # 3. If there is a ToolMessage in history that hasn't been graded, go to grader
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            return "grader"
    
    return "end"



def route_after_grading(state: State):
    # Get the last message, which is the grader's output
    grade_content = state["messages"][-1].content.strip().lower()
    
    # Simple check for 'yes'
    if "yes" in grade_content:
        return "end"
    
    # Anything else (including 'no') routes back to assistant
    return "assistant"

# ==========================================
# 4. Build Graph
# ==========================================
builder = StateGraph(State)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(all_rag_tools))
builder.add_node("grader", grader_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant", 
    route_assistant,
    {
        "tools": "tools",         # Map function return "tools" -> "tools" node
        "assistant": "assistant", # Map function return "assistant" -> "assistant" node
        "end": END,               # Map function return "end" -> END constant
        "grader": "grader"        # ADD THIS LINE: Map function return "grader" -> "grader" node
    }
)



builder.add_edge("tools", "assistant")


# Change this specific line in your Builder section
builder.add_conditional_edges(
    "grader", 
    route_after_grading, 
    {"assistant": "assistant", "end": END} 
)

rag_graph_agent = builder.compile()

# ==========================================
# 5. Execution Interface
# ==========================================

# In rag_agent.py

def get_state_logs(result):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print("\n===== STATE LOGS =====")
        
        # Access the retry count directly from the result dictionary
        retries = result.get("grader_retries", 0)
        print(f"DEBUG: Current Grader Retries: {retries}")
        
        for i, msg in enumerate(result["messages"]):
            # Use msg.content or a fallback if content is empty
            content = getattr(msg, "content", "No content")
            print(f"[{i}] {msg.__class__.__name__}: {content}")
            
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"    Tool Calls: {msg.tool_calls}")
        print("======================\n")
    return output.getvalue()



def ask_legal_agent(question: str):
    result = rag_graph_agent.invoke({"messages": [("user", question)], "grader_retries": 0})
    
    # Get the logs string
    logs = get_state_logs(result)
    
    # Find the answer
    answer = "No answer generated."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and len(msg.content) > 5:
            answer = msg.content
            break
            
    return answer, logs # Return both the answer and the logs




# if __name__ == "__main__":
#    user_query = "Is my right to equality protected from constitution."
#    answer = ask_legal_agent(user_query)
#    print(f"\nFinal Answer: {answer}")