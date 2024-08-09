from typing import Dict
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config import get_env_variable

@tool
def dummy_tool(expression: str) -> str:
    """Evaluate expressions; placeholder tool."""
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

def create_agent(role: str, llm: ChatOpenAI) -> AgentExecutor:
    """Create and return an agent for a given role."""
    system_prompts = {
        "Outline Writer": "Create a detailed outline with plot points, character arcs, and setting descriptions.",
        "Character Designer": "Design characters with detailed traits, emotional nuances, and appearances.",
        "Environment Designer": "Design immersive settings with physical descriptions, atmosphere, and mood.",
        "Script Writer": "Combine all elements into a cohesive script. Manage and direct other agents for necessary information."
    }
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompts[role]),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    tools = [dummy_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

class SimpleAgent:
    def __init__(self, role: str, executor: AgentExecutor):
        self.role = role
        self.executor = executor

    def perform_task(self, state) -> str:
        """Perform the agent's task and handle errors."""
        try:
            assert 'messages' in state.state, "Missing 'messages' in state"
            result = self.executor.invoke(state.state)
            return result["output"].strip()
        except Exception as e:
            print(f"Error for {self.role}: {e}")
            raise

def setup_agents() -> Dict[str, SimpleAgent]:
    """Setup and return the agents."""
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    roles = ["Outline Writer", "Character Designer", "Environment Designer", "Script Writer"]
    agents = {role: SimpleAgent(role, create_agent(role, llm)) for role in roles}
    return agents