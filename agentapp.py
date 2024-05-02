# General imports
import streamlit as st
from streamlit_chat import message
import os
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
import openai

# Langchain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import AgentExecutor, create_openai_tools_agent,load_tools
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

## Additional Library - 
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader

headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json",
    "talivy_api_key":st.secrets['TAVILY_API_KEY'],
    "langchain_api_key":st.secrets['LANGCHAIN_API_KEY'],
    "openweathermap_api_key":st.secrets['OPENWEATHERMAP_API_KEY'],
    "e2b_api_key":st.secrets['E2B_API_KEY'],
    "serpapi_api_key":st.secrets['SERPAPI_API_KEY']
    }

openai.api_key = st.secrets["OPENAI_API_KEY"]
OPENWEATHERMAP_API_KEY = st.secrets['OPENWEATHERMAP_API_KEY']

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['OPENWEATHERMAP_API_KEY'] = st.secrets['OPENWEATHERMAP_API_KEY']
os.environ['E2B_API_KEY'] = st.secrets['E2B_API_KEY']
os.environ['SERPAPI_API_KEY'] = st.secrets['SERPAPI_API_KEY']

st.title('Multi-Agent Framework - Supervisor')
st.write(""":blue[Following agents are used in this framwork: Researcher, Coder, Weather, Calculator, Youtuber,
          GoogleFinace, JobSearch, Wikipedia, WebScarp, WebSummerizer]""")
# system_p")
# For visualization of the graph, Optional
## Creating LLM
## Tools and llm setup
llm = ChatOpenAI(model="gpt-4-1106-preview")

# Weather tools
weather_tool = load_tools(["openweathermap-api"], llm, openweathermap_api_key=OPENWEATHERMAP_API_KEY)

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
youtube_tool = YouTubeSearchTool()

## Google Finace
google_finance_tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())


## Google Job Search
jobsearch_tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())


# Dall.E image
dalle_tool = DallEAPIWrapper()

## Wikipedia
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# response=wikipedia.run("Barak Obama")
# response

# Web Scrape

@tool()
def webScarpe(urls: List[str]) -> str:
  """Use requests and bs4 to scrape the provided web pages for detailed 
     information."""
  loader = WebBaseLoader(urls)
  docs = loader.load()
  return "\n\n".join(
      [
          f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
          for doc in docs
      ]
  )

@tool()
def summarize(content: str) -> str:
    """Summarize the content"""
    messages = [
    SystemMessage(
        content="You are an AI agent having very borad knowledge about the recent events around the glob. Your task is to summarize the given content very accurately."
        ),
        HumanMessage(
        content=content
        ),
    ]
    response = llm(messages)
    return response.content

#cnt=webScarpe.invoke({'urls':['https://blog.langchain.dev/reflection-agents/']})

@tool()
def YoutubeTranslate(url: str) -> str:
    """Use to tranlate youtube video to plain text"""
    loader = YoutubeLoader.from_youtube_url(url,
    add_video_info=True,
    language=["en", "id"],
    translation="en")

    txt=loader.load()

    return txt[0].page_content


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

members = ["Researcher", "Coder", "Weather", "Calculator", "Youtuber", "GoogleFinace", "JobSearch", "Wikipedia", "WebScarp", "WebSummerizer"]# "YouTubeText"]
# system_prompt = (
#     "You are a supervisor tasked with managing a conversation between the"
#     " following workers:  {members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,"
#     " respond with FINISH."
# )
system_prompt = (
    "You are the supervisor over the following agents: {members}."
    "You are responsible for assigning tasks to each agent as requested by the user."
    "Each agent executes tasks according to their roles and responds with their results and status."
    "Please review the information and answer with the name of the agent to which the task should be assigned next."
    "Answer 'FINISH' if you are satisfied that you have fulfilled the user's request."
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


# image_generator_agent  = create_agent(
#     llm,
#     [dalle_tool],
#     "You are helpful for generating images"
# )
# image_generator_node = functools.partial(agent_node, agent=image_generator_agent, name="ImageGenerator")


WebScarp_agent  = create_agent(
    llm,
    [webScarpe],
    "You are helpful for scarping the web page"
)
WebScarp_node = functools.partial(agent_node, agent=WebScarp_agent, name="WebScarp")

WebSummerizer_agent = create_agent(
    llm,
    [summarize],
    "You are helpful for summarizing the content"
)
WebSummerizer_node = functools.partial(agent_node, agent=WebScarp_agent, name="WebSummerizer")

YoutubeTranslate_agent = create_agent(
    llm,
    [YoutubeTranslate],
    "You are helpful for translating audio/video to text"
)
YoutubeTranslate_node = functools.partial(agent_node, agent=WebScarp_agent, name="YouTubeText")

#### Create a Graph 
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Weather", weather_node)
workflow.add_node("Calculator", math_node)
workflow.add_node("Youtuber", youtube_node)
workflow.add_node("GoogleFinace", google_finance_node)
workflow.add_node("JobSearch", jobsearch_node)
workflow.add_node("Wikipedia", wiki_search_node)
workflow.add_node("WebScarp", WebScarp_node)
workflow.add_node("WebSummerizer", WebSummerizer_node)
#workflow.add_node("YouTubeText", YoutubeTranslate_node)
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

user_query = "find the weather in New York City today"
user_query = "weather in ST. Louis MO and find one job related to data science in St. Louis?"
user_query = "what is the google's stock today?"
# user_query = "Why students are protesting in Columbia Univeristy this year?"

user_query="1. First extract content from this url=https://python.langchain.com/docs/modules/agents/quick_start/, and 2. Summarize the content with one sentence only"

# user_query="get the content from this url=https://www.youtube.com/watch?v=BZP1rYjoBgI" # this first goes to web and then go to youtube

#user_query="write a python function help me generate Fibonacci sequence "


question = st.text_input("User Query", key="input")
#question = "find the weather in New York City today"

if st.button('Submit'):
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
                if len(state) > 3:
                    break
                    print(len(state))
        except:
            print("There is an issue")

        if len(state) > 1:   

            #key_name=list(state[-1].keys())[0]

            #content=state[-1][key_name]['messages'][0].content

            #final_agent=state[-1][key_name]['messages'][0].name

            #st.write(f":blue[{content}]")
            # st.write(f"Final Agent: {final_agent}")
            st.write(":red[Let's check the entire state]")
            st.write(state)
            # st.write("Thank you for using this demo")
        else:
            st.write("service is not available currenlty")
