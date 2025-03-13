import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from io import StringIO
from typing_extensions import TypedDict
from typing import Annotated, Any
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolNode
import matplotlib.pyplot as plt
import streamlit as st


OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
# DB and LLM Setup
db = SQLDatabase.from_uri("sqlite:///mydb.db")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY, temperature = 0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()



# State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
llm_to_get_schema = llm.bind_tools([get_schema_tool])

# Tools
from langchain_core.tools import tool

@tool
def query_to_database(query: str) -> str:
    """
    Execute a SQL query against the database and return the result.
    If the query is invalid or returns no result, an error message will be returned.
    """    
    result = db.run_no_throw(query)
    return result if result else "Error: Query failed."

def execute_query(state: State):
    """
    Execute a SQL query against the database and return the result.
    If the query is invalid or returns no result, an error message will be returned.
    """   
    sql_query = state["messages"][-1].content
    result = db.run_no_throw(sql_query)

    if not result:
        result = "Error: Query execution returned no data."

    return {"messages": [AIMessage(content=result)]}


# Tool Nodes
def handle_tool_error(state: Any):
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(content=f"Error: {repr(error)}", tool_call_id=tc["id"])
            for tc in tool_calls
        ]
    }

def create_tool_node(tools: list):
    print(tools)
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

list_tables = create_tool_node([list_tables_tool])
get_schema = create_tool_node([get_schema_tool])
query_database = create_tool_node([query_to_database])


# State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Prompts
query_gen_system_prompt = """Generate ONLY a valid SQLite SQL query. Do NOT invoke any tools."""

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", query_gen_system_prompt),
    ("placeholder", "{messages}")
])

query_generator = query_gen_prompt | llm

# Nodes
def first_tool_call(state: State):
    return {"messages": [AIMessage(content="", tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": "1"}])]}

def llm_get_schema(state: State):
    response = llm_to_get_schema.invoke(state["messages"])
    return {"messages": [response]}

def generation_query(state: State):
    message = query_generator.invoke(state)
    return {"messages": [message]}

def check_the_given_query(state: State):
    return {"messages": [state["messages"][-1]]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "execute_query"
    if last_message.content.startswith("Error:"):
        return "query_gen"
    return "correct_query"

def get_final_execution(sql_query, user_query):
    query_data = query_to_database(sql_query.replace("```sql", "").replace("```",""))

    # print("query_output ----> ",query_data)

    extraction_prompt = f"""
    You are an AI data analyst.
    Given to you are sql_query : {sql_query}
    Queried output from DB : {query_data}
    User given Query: {user_query}

    Your task is to explain in about the data fetched in brief. Use the Quried output {query_data} from DB as the output.
    *** Give me the explaination in Bullets***

    ***Strictly don't include the SQL Queries as the output
    """

    extraction_response = llm.invoke(extraction_prompt).content.strip()
    return extraction_response

# def visualization_tool(state: State):
#     """
#     Extract data and plot type from query results and user input.
#     Generates and executes matplotlib code via GPT.
#     Returns both the executed query result and visualization status.
#     """

#     import pandas as pd
#     from io import StringIO
#     import matplotlib.pyplot as plt

#     user_query = state["messages"][0].content
#     query_output = state["messages"][-1].content

#     print("query_output", query_output)

#     query_data = query_to_database(query_output.replace("```sql", "").replace("```",""))
#     user_query_prompt= f"""
#     You are my AI Assistant. Check if the below user query has a request to plot a graph or not.
#     If yes return True or False
#     User Query: {user_query}
#     """

#     user_query_response = llm.invoke(user_query_prompt).content.strip()

#     if (user_query_response == "False"):
#         full_message_content = f"Executed Query Result:\n{get_final_execution(query_output, user_query)}"
#         return {"messages": [ToolMessage(content=full_message_content, tool_call_id="visualization_tool")]}

#     final_queried_output = get_final_execution(query_output, user_query)
#     print(f"Queries Output :=\n--------\n{final_queried_output}\n---------\n")
    
#     extraction_prompt = f"""
#     You are an AI data analyst. Convert the query result into CSV format and identify the plot type from the user's query.

#     Query Output: {query_data}
#     User Query: {user_query}

#     Format exactly as:
#     CSV:
#     column1,column2
#     val1,val2

#     Plot Type: plot_type
#     """

#     extraction_response = llm.invoke(extraction_prompt).content.strip()

#     # Robust parsing logic:
#     try:
#         if "\n\nPlot Type: " in extraction_response:
#             csv_data, plot_type = extraction_response.split("\n\nPlot Type: ")
#         elif "Plot Type:" in extraction_response:
#             csv_data, plot_type = extraction_response.split("Plot Type:")
#         else:
#             raise ValueError("GPT response format incorrect. 'Plot Type' not found.")
#         csv_data = csv_data.replace("CSV:\n", "").strip()
#         plot_type = plot_type.strip()
#         df = pd.read_csv(StringIO(csv_data))
#     except Exception as e:
#         return {
#             "messages": [ToolMessage(
#                 content=f"Data extraction failed: {e}\nGPT response: {extraction_response}",
#                 tool_call_id="visualization_tool")]
#         }

#     plot_prompt = f"""
#     Generate matplotlib Python code for a '{plot_type}' plot using this CSV data:
#     {df.to_csv(index=False)}

#     Provide ONLY the Python code without markdown.
#     """

#     plot_code_response = llm.invoke(plot_prompt).content.strip()

#     # REMOVE MARKDOWN FORMATTING CLEARLY
#     if plot_code_response.startswith("```python"):
#         plot_code_response = plot_code_response[len("```python"):].strip()
#     if plot_code_response.endswith("```"):
#         plot_code_response = plot_code_response[:-len("```")].strip()

#     # Save and execute visualization script
#     os.makedirs(".Images", exist_ok=True)
#     script_path = "plot.py"
#     with open(script_path, "w") as f:
#         f.write(plot_code_response)

#     try:
#         exec(plot_code_response, {"pd": pd, "plt": plt})
#         visualization_status = f"Visualization successful. Script saved at {script_path}"
#     except Exception as e:
#         visualization_status = f"Visualization failed: {e}\nGenerated Code:\n{plot_code_response}"

#     # Combine executed query output and visualization status clearly
    
#     full_message_content = f"Executed Query Result:\n{final_queried_output}\n\n{visualization_status}"

#     return {"messages": [ToolMessage(content=full_message_content, tool_call_id="visualization_tool")]}

def visualization_tool(state: State):
    """
    Extract data and plot type from query results and user input.
    Generates and executes matplotlib code via GPT.
    Returns both the executed query result and visualization status.
    """

    import pandas as pd
    from io import StringIO
    import matplotlib.pyplot as plt
    import subprocess

    user_query = state["messages"][0].content
    query_output = state["messages"][-1].content

    print("query_output", query_output)

    query_data = query_to_database(query_output.replace("```sql", "").replace("```",""))
    user_query_prompt= f"""
    You are my AI Assistant. Check if the below user query has a request to plot a graph or not.
    If yes return True or False
    User Query: {user_query}
    """

    user_query_response = llm.invoke(user_query_prompt).content.strip()

    if (user_query_response == "False"):
        full_message_content = f"Executed Query Result:\n{get_final_execution(query_output, user_query)}"
        return {"messages": [ToolMessage(content=full_message_content, tool_call_id="visualization_tool")]}

    final_queried_output = get_final_execution(query_output, user_query)
    print(f"Queries Output :=\n--------\n{final_queried_output}\n---------\n")
    
    extraction_prompt = f"""
    You are an AI data analyst. Convert the query result into CSV format and identify the plot type from the user's query.

    Query Output: {query_data}
    User Query: {user_query}

    Format exactly as:
    CSV:
    column1,column2
    val1,val2

    Plot Type: plot_type
    """

    extraction_response = llm.invoke(extraction_prompt).content.strip()

    # Robust parsing logic:
    try:
        if "\n\nPlot Type: " in extraction_response:
            csv_data, plot_type = extraction_response.split("\n\nPlot Type: ")
        elif "Plot Type:" in extraction_response:
            csv_data, plot_type = extraction_response.split("Plot Type:")
        else:
            raise ValueError("GPT response format incorrect. 'Plot Type' not found.")
        csv_data = csv_data.replace("CSV:\n", "").strip()
        plot_type = plot_type.strip()
        df = pd.read_csv(StringIO(csv_data))
    except Exception as e:
        return {
            "messages": [ToolMessage(
                content=f"Data extraction failed: {e}\nGPT response: {extraction_response}",
                tool_call_id="visualization_tool")]
        }

    plot_prompt = f"""
    Generate matplotlib Python code for a '{plot_type}' plot using this CSV data:
    {df.to_csv(index=False)}

    Provide ONLY the Python code without markdown.
    """

    plot_code_response = llm.invoke(plot_prompt).content.strip()

    # REMOVE MARKDOWN FORMATTING CLEARLY
    if plot_code_response.startswith("```python"):
        plot_code_response = plot_code_response[len("```python"):].strip()
    if plot_code_response.endswith("```"):
        plot_code_response = plot_code_response[:-len("```")].strip()

    # Ensure the generated Matplotlib code saves the plot
    save_plot_code = "\nplt.savefig('generated_plot.png', bbox_inches='tight')\n"

    # Append the save code to the generated code
    plot_code_response += save_plot_code

    # Save and execute visualization script
    os.makedirs(".Images", exist_ok=True)
    script_path = "plot.py"
    with open(script_path, "w") as f:
        f.write(plot_code_response)

    try:
        subprocess.run(["python", script_path], check=True)  # Run plot.py
        visualization_status = f"Visualization successful. Script saved at {script_path}"
    except Exception as e:
        visualization_status = f"Visualization failed: {e}\nGenerated Code:\n{plot_code_response}"

    # Combine executed query output and visualization status clearly
    
    full_message_content = f"{final_queried_output}"

    return {"messages": [ToolMessage(content=full_message_content, tool_call_id="visualization_tool")]}


# Workflow
workflow = StateGraph(State)
workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool", list_tables)
workflow.add_node("model_get_schema", llm_get_schema)
workflow.add_node("get_schema_tool", get_schema)
workflow.add_node("query_gen", generation_query)
workflow.add_node("correct_query", check_the_given_query)
workflow.add_node("execute_query", query_database)
workflow.add_node("visualization", visualization_tool)

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")

workflow.add_conditional_edges("query_gen", should_continue, {
    END: END,
    # "query_gen": "query_gen",
    "correct_query": "correct_query",
    "execute_query": "execute_query"
})

workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "visualization")
workflow.add_edge("visualization", END)


app = workflow.compile()

# Display workflow graph
# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod

# display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))
