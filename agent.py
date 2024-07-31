import openai
import os
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.smith import RunEvalConfig,run_on_dataset
from langsmith import Client 
from dotenv import load_dotenv

load_dotenv()
# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-storywriter"

#Langsmith for tracing
client = Client()

def create_agent(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful story writer."),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    tools = [dummy_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

@tool
def dummy_tool(expression: str) -> str:
    """You do not need to use any tools. Just write the response yourself."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)


# Define a simple agent class
class SimpleAgent:
    def __init__(self, executor):
        self.executor = executor

    def generate_story(self, prompt):
        messages = [{"role": "user", "content": f"Generate a story based on the following prompt: {prompt}"}]
        state = {"messages": messages, "agent_scratchpad": []}
        result = self.executor.invoke(state)
        return result["output"].strip()


# Initialize the agent
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
executor = create_agent(llm)
agent = SimpleAgent(executor=executor)

def generate_story_from_prompt(prompt: str) -> str:
    return agent.generate_story(prompt)