import openai
import os
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-storywriter"

client = Client()

@tool
def dummy_tool(expression: str) -> str:
    """You do not need to use any tools. Just write the response yourself."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)

def create_agent(role: str, llm: ChatOpenAI):
    system_prompts = {
        "Outline Writer": "You are an outline writer for a written story. You develop a comprehensive outline that captures the main plot points, character arcs, and setting descriptions. Ensure that the structure supports a clear narrative flow and aligns with the overall themes and settings of the story. Your outline should provide a detailed framework that guides the development of the narrative. Be sure to be in line with the overall themes and tone of the story.",
        "Character Designer": "You are a character designer for a written story. You create distinct and expressive character designs, ensuring each reflects its unique personality and backstory. Incorporate traits and emotional nuances that align with the overall story and setting. The visual appearance should cohesively represent the character’s inner world and their role in the narrative. Be sure to be in line with the overall themes and tone of the story.",
        "Environment Designer": "You are an environment designer. Create detailed and immersive settings and locations that match the story's needs, including thorough physical descriptions, atmosphere, and mood. Ensure that these environments are vivid and contribute effectively to the storytelling, enhancing the overall narrative impact. Be sure to be in line with the overall themes and tone of the story.",
        "Script Writer": "You are a script writer. You consolidate all elements of the story, including the outline, characters, and environments, into a cohesive narrative. You will manage and direct other agents, such as the Outline Writer, Character Designer, and Environment Designer, to gather necessary information. Deliver a well-structured script that effectively captures the essence and progression of the story, ensuring it resonates with the intended audience."
    }
    system_prompt = system_prompts[role]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    tools = [dummy_tool]
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

class SimpleAgent:
    def __init__(self, role, executor):
        self.role = role
        self.executor = executor

    def perform_task(self, state):
        try:
            assert 'messages' in state.state, "State is missing 'messages'"
            result = self.executor.invoke(state.state)
            return result["output"].strip()
        except Exception as e:
            print(f"Error performing task for role {self.role}: {str(e)}")
            raise e

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
roles = ["Outline Writer", "Character Designer", "Environment Designer", "Script Writer"]
agents = {role: SimpleAgent(role, create_agent(role, llm)) for role in roles}

class AgentState:
    def __init__(self, messages, next_agent):
        self.state = {
            "messages": messages,
            "next_agent": next_agent
        }

    def __getitem__(self, key):
        if key not in self.state:
            raise KeyError(f"Key {key} not found in state")
        return self.state[key]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __contains__(self, key):
        return key in self.state

class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

    def get_next_node(self, current_node):
        return self.edges.get(current_node, [])

def outline_writer_node(state):
    return agents["Outline Writer"].perform_task(state)

def character_designer_node(state):
    return agents["Character Designer"].perform_task(state)

def environment_designer_node(state):
    return agents["Environment Designer"].perform_task(state)

def script_writer_node(state):
    return agents["Script Writer"].perform_task(state)

state_graph = StateGraph()
state_graph.add_node("Outline Writer", outline_writer_node)
state_graph.add_node("Character Designer", character_designer_node)
state_graph.add_node("Environment Designer", environment_designer_node)
state_graph.add_node("Script Writer", script_writer_node)

state_graph.add_edge("Script Writer", "Outline Writer")
state_graph.add_edge("Script Writer", "Character Designer")
state_graph.add_edge("Script Writer", "Environment Designer")

class Director:
    def __init__(self, state_graph):
        self.state_graph = state_graph
        self.current_node = "Script Writer"

    def run_workflow(self, initial_prompt):
        state = AgentState(messages=[{"role": "user", "content": initial_prompt}], next_agent=self.current_node)
        while state["next_agent"]:
            current_func = self.state_graph.nodes[state["next_agent"]]
            output = current_func(state)
            print(f"Output from {state['next_agent']}: {output}")  # debugging
            state["messages"].append({"role": "assistant", "content": output})
            next_nodes = self.state_graph.get_next_node(state["next_agent"])
            if next_nodes:
                state["next_agent"] = next_nodes[0]
            else:
                state["next_agent"] = None
        return state["messages"][-1]["content"]

def generate_story_from_prompt(prompt: str) -> str:
    director = Director(state_graph)
    return director.run_workflow(prompt)

if __name__ == "__main__":
    prompt = "A hero's journey in a mystical land."
    story = generate_story_from_prompt(prompt)
    print("Final Story:")
    print(story)
