import os
from langchain.chains import LLMMathChain
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Sequence
from langgraph.graph import END, MessageGraph
import re
import streamlit as st

st.title("Solving Math Problem with LLM")

## Sample Questions
question = "The greatest common divisor of positive integers m and n is 6.The least common multiple of m and n is 126. What is the leastpossible value of m + n?"

question="The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more,how many apples do they have?"

question="Today is 27 Feb, 2023. I was born exactly 35 years ago. What is the date I was born in MM/DD/YYYY"



question = st.text_area("Provide your question here")




OPENAI_API_KEY="xxxx" # https://platform.openai.com/account/api-keys
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = ChatOpenAI(model='gpt-4')


######################################################## Generator ###########################
#Define agent for generating code
#Define PythonREPLTool

tools = [PythonREPLTool()]


template="""

{instructions}

TOOLS:

------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

messages: {messages}

{agent_scratchpad}

"""

base_prompt = PromptTemplate.from_template(template)


instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""
#base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt_agent = base_prompt.partial(instructions=instructions)
agent = create_react_agent(llm, tools, prompt_agent)

#generate = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

generate = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)




####################### Reflection #########################################

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very good at understanding mathematical reasoning and you are a python code evaluator."
            "You should check the provided python code whether it is consistent with the given query or not."
            "You only check the code. You won't write the new code."
            "If the code is consistent with the user query, should answer yes only. Otherwise, you should answer no with reason why it is not matching.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm


############################### Build LangGraph #######################################

def generation_node(state: Sequence[BaseMessage]):
    # print('----')
    # print(state)
    result_generate = generate.invoke({"messages": state})
    return AIMessage(content="Here is the python code: " +result_generate['intermediate_steps'][-1][0].tool_input + "and final output is: " +  result_generate['output'])


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # print('------')
    # print(messages)
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = reflect.invoke({"messages": translated})
    # We treat the output of this as human feedback for the generator
    return HumanMessage(content=res.content)

builder = MessageGraph()
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.set_entry_point("generate")

def should_continue(state):
    last_message = state[-1]
    # If there is no function call, then we finish
    if "yes"  in last_message.content.lower() or len(state) > 6:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

builder.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "continue":"generate",
        "end": END
    }
)

builder.add_edge("generate", "reflect")
#builder.add_edge("generate", "reflect")

graph = builder.compile()

################################ output format ####################################
def extract_python_code(text):
    # Define a regular expression pattern to find Python code blocks
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        # Return the extracted code block without the backticks
        return match.group(1).strip()
    else:
        # Return a message if no code block is found
        return "No Python code block found."

def extract_final_output(text):
    # Search for the starting point of the phrase "final output is:"
    start_index = text.find("final output is:")
    
    if start_index == -1:
        # Phrase not found, return a failure message
        return "Calculation fails."
    
    # Extract and return the text following "final output is:"
    # Adding the length of "final output is:" to start_index and trimming any leading spaces or colons
    output_text = text[start_index + len("final output is:"):].strip()
    return output_text

if st.button('Submit'):
    request =HumanMessage(question)
    for event in graph.stream(
        [
            request
        ],
    ):  
        print('...', end='')
    
    final_output=event['__end__'][-2].content

    python_code = extract_python_code(final_output)

    result_final=extract_final_output(final_output)

    st.write("Python Code  Generated by an Agent")
    st.write("\n")
    st.code(python_code, language="python")
    st.write('----------------------------------')
    st.write("final result")
    st.write("\n")
    st.write(result_final)
    st.write('----------------------------------')
    st.write("All steps taken by reflection agent")
    st.write('\n')
    st.write(event)