import os
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import streamlit as st
from streamlit_chat import message

# Langchain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import AgentExecutor, create_openai_tools_agent,load_tools
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENWEATHERMAP_API_KEY'] = OPENWEATHERMAP_API_KEY
os.environ['E2B_API_KEY'] = E2B_API_KEY
os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

st.title('Multi-Agent Framework Controlled by Supervisor')

image_path = 'multiagent.png'  # Adjust the path to your image file
st.image(image_path, caption='Multi-Agent Framework', width=800)

# Optional, add tracing in LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Reflection"

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List

# For visualization of the graph, Optional
from IPython.display import Image
import pygraphviz
from IPython.display import Image


## Creating LLM

## Tools and llm setup
llm = ChatOpenAI(model="gpt-4-1106-preview")

# Weather tools
weather_tool = load_tools(["openweathermap-api"], llm)

# Tavily tools - Search tools
tavily_tool = TavilySearchResults(max_results=5)

### Math tools
problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(name="Calculator",func=problem_chain.run,description=\
                               "Useful for when you need to answer questions \
about math. This tool is only for math questions and nothing else. Only input\
math expressions.")

# Python REPL tool 
python_repl_tool = PythonREPLTool()

## Youtube tool
from langchain_community.tools import YouTubeSearchTool
youtube_tool = YouTubeSearchTool()

## Google Finace
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
google_finance_tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())


## Google Job Search
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
jobsearch_tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())


# Dall.E image
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
dalle_tool = DallEAPIWrapper()

## Wikipedia
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# response=wikipedia.run("Barak Obama")
# response

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Researcher", "Coder", "Weather", "Calculator", "Youtuber", "GoogleFinace", "JobSearch", "Wikipedia"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

### Supervisor  Chain
supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    recursion_count: int = 0

def check_recursion(state):
    if state.recursion_count >= 1:  # Limit set to 1
        return "END"
    state.recursion_count += 1
    return "CONTINUE"


research_agent = create_agent(llm, [tavily_tool], "You are a Web searcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

#Weather nodes 
weather_agent = create_agent(
    llm,
    weather_tool,
    "You are a weather finder. You can provide weather information for a given location.",
)
weather_node = functools.partial(agent_node, agent=weather_agent, name="Weather")

# Math node

math_agent = create_agent(
    llm,
    [math_tool],
    "Useful for when you need to answer questions about math",
)
math_node = functools.partial(agent_node, agent=math_agent, name="Calculator")

youtube_agent  = create_agent(
    llm,
    [youtube_tool],
    "You are helpful for finding videos from Youtube"
)
youtube_node = functools.partial(agent_node, agent=youtube_agent, name="Youtuber")

google_finance_agent  = create_agent(
    llm,
    [google_finance_tool],
    "You are helpful for finding videos from Youtube"
)
google_finance_node = functools.partial(agent_node, agent=google_finance_agent, name="GoogleFinace")


jobsearch_agent  = create_agent(
    llm,
    [jobsearch_tool],
    "You are helpful for finding job related information"
)
jobsearch_node = functools.partial(agent_node, agent=jobsearch_agent, name="JobSearch")


wiki_search_agent  = create_agent(
    llm,
    [wiki_tool],
    "You are helpful for finding information from Wikipedia"
)
wiki_search_node = functools.partial(agent_node, agent=jobsearch_agent, name="Wikipedia")


workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Weather", weather_node)
workflow.add_node("Calculator", math_node)
workflow.add_node("Youtuber", youtube_node)
workflow.add_node("GoogleFinace", google_finance_node)
workflow.add_node("JobSearch", jobsearch_node)
workflow.add_node("Wikipedia", wiki_search_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")
graph = workflow.compile()

# Graph Visualization
# os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'
# Image(graph.get_graph().draw_png())


# user_query = "find the weather in New York City today"
# user_query = "weather in ST. Louis MO and find one job related to data science in St. Louis?"
# #user_query = "what is the google's stock today?"
# user_query = "Why students are protesting in Columbia Univeristy this year?"

question = st.text_input("User question", key="input")
#question = "find the weather in New York City today"


if question:
    state=[]
    try:
        for s in graph.stream(
            {
                "messages": [
                    HumanMessage(content=f"{question}")
                ]
            }
        ):
            state.append(s)
            if "__end__" not in s:
                #st.write(s)
                print("----")
            if len(state) > 5:
                break
                print(len(state))
    except:
        print("There is an issue")

    if len(state) > 1:   

        key_name=list(state[-1].keys())[0]

        content=state[-1][key_name]['messages'][0].content

        final_agent=state[-1][key_name]['messages'][0].name

        st.write(content)
        st.write(f"Final Agent: {final_agent}")
        st.write("Check the entire state")
        st.write(state)
        st.write("Thank you for using this demo")
    else:
        st.write("service is not available currenlty")


