from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import requests
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# define llm
model = ChatOpenAI(model="gpt-4o")

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# define tools
class City(BaseModel):
    city: str = Field(description="City")
    country: str = Field(description="Country code")

def get_current_weather(city: str, country: str) -> int:
    response = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={OPENWEATHERMAP_API_KEY}"
    )
    data = response.json()
    temp_kelvin = data["main"]["temp"]
    temp_fahrenheit = (temp_kelvin - 273.15) * 9 / 5 + 32
    return int(temp_fahrenheit)


weather = StructuredTool.from_function(
    func=get_current_weather,
    name="Get_Weather",
    description="Get the current temperature from a city, in Fahrenheit",
    args_schema=City,
    return_direct=False,
)

tools = [weather, TavilySearchResults(max_results=2)]
graph = create_react_agent(model, tools=tools)
