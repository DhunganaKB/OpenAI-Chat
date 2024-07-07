import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_experimental.tools import PythonREPLTool
import re
from langchain_core.runnables import chain


OPENAI_API_KEY="sk-proj-xxxxxxxxxx" # https://platform.openai.com/account/api-keys
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
llm = ChatOpenAI(model='gpt-4')

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

When you have a response to say to the Human

```

Final Answer: [your response here]

```

Begin!

messages: {messages}

{agent_scratchpad}

"""

base_prompt = PromptTemplate.from_template(template)

tools = [PythonREPLTool()]

instructions = """You are an agent designed to print and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question.
Unfortunately, the Assistant is terrible at maths. When provided with math questions,
no matter how simple, you should always refer to its trusty tools and absolutely does NOT try to answer math questions by yerself.
You have to make sure, tools should be always used for all kind of math problem
"""
#base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt_agent = base_prompt.partial(instructions=instructions)
agent = create_react_agent(llm, tools, prompt_agent)

generate = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,
                          handle_parsing_errors=True, force_tool=True)


def extract_python_code(text):
    # Define a regular expression pattern to find Python code blocks
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    pattern1 = r"```(.*?)```"
    match = pattern.search(text)
    match1 = re.search(pattern1, text, re.DOTALL)
    
    if match:
        # Return the extracted code block without the backticks
        return match.group(1).strip()
    elif match1:
        return match1.group(1).strip()
    else:
        # Return a message if no code block is found
        return text
        #return "No Python code block found."

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



def code_gen_run(question):
    # Simulate invoking an external process (replace with your actual implementation)
    output = generate.invoke({"messages": question})
    final_output = {
        'input': question,
        'output': output['output'],
        'code': extract_python_code(output['intermediate_steps'][-1][0].tool_input)
    }
    return final_output

def streamlit_app():
    st.title("Python Code Execution Agent")
    question = st.text_input("Enter your question:", key="question")
    
    if "approved" not in st.session_state:
        st.session_state.approved = None

    if st.button("Generate Code"):
        st.session_state.original_msg = code_gen_run(question)
    
    if "original_msg" in st.session_state:
        original_result, original_code = st.session_state.original_msg['output'], st.session_state.original_msg['code']
        st.subheader("Original Output:")
        st.write(original_result)
        st.subheader("Original Python Code:")
        st.code(original_code, language="python")

        st.session_state.approved = st.radio("Do you accept the solution provided?", ('Yes', 'No'))

    if st.session_state.approved == 'No':
        regenerated_msg = code_gen_run(question)
        regenerated_result, regenerated_code = regenerated_msg['output'], regenerated_msg['code']
        
        st.subheader("Regenerated Output:")
        st.write(regenerated_result)
        st.subheader("Regenerated Python Code:")
        st.code(regenerated_code, language="python")

if __name__ == "__main__":
    streamlit_app()