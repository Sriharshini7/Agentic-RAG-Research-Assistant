from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from state import State
from dotenv import load_dotenv
import os


load_dotenv()

# Get the API key explicitly from the environment variables
api_key = os.getenv("GROQ_API_KEY")

#==================================
# 1. LLM Brain and Bindings Setup
#==================================

grader_llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0,
    api_key = api_key, # Cleaned up to use the variable we extracted above
)


GRADER_SYSTEM_MESSAGE = """
"You are a grader. Evaluate if the retrieved context is relevant to the user query. 
If the context provides at least some foundational information that can be used to infer an answer, output 'Yes'."
If it it irrelevant or insufficient to answer, output 'No'.
You response must be a single word: 'Yes' or 'No'.
"""

MAX_GRADER_RETRIES = 3




def grader_node(state:State):
    """Evaluates if the retrieved context is sufficient."""
    retries = state.get("grader_retries", 0) + 1

    # If we've exceeded max retries, force a pass to avoid infinite loops
    if retries > MAX_GRADER_RETRIES:
        return {"messages": ["Yes"], "grader_retries": retries}

    #'messages' is a list containing all interactions so far
    message = state["messages"]
    #[0] is the very first message: the user's original query
    question = message[0].content

    #looking for the latest message
    context = ""
    for msg in reversed(message):
        if isinstance(msg,ToolMessage):
            context = msg.content
            break
    #if no output from tool
    if not context:
        return {"messages": ["No"], "grader_retries": retries}
    
    #sending both original retrieved context to LLM 
    grade_response = grader_llm.invoke([SystemMessage(content=GRADER_SYSTEM_MESSAGE)] + 
                        [f"Query: {question}\n\nContext:{context}"])
    
    #return yes or no
    return {"messages": [grade_response], "grader_retries": retries}

