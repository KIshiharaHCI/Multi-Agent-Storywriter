from typing import Dict, Callable

class AgentState:
    def __init__(self, messages, next_agent: str, story_parts: Dict[str, str]):
        self.state = {"messages": messages, "next_agent": next_agent, "story_parts": story_parts}

    def __getitem__(self, key: str):
        if key not in self.state:
            raise KeyError(f"Key {key} not found")
        return self.state[key]

    def __setitem__(self, key: str, value):
        self.state[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.state

    def add_story_part(self, part_name: str, content: str):
        self.state["story_parts"][part_name] = content

    def get_story_content(self) -> str:
        """Concatenate all story parts."""
        return "\n\n".join(self.state["story_parts"].values())

class StateGraph:
    """Manages nodes and edges in the state transition graph.
    Nodes are agents and edges the communication between agents."""
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

def script_writer_node(state: AgentState, agents: Dict[str, Callable[[AgentState], str]]) -> AgentState:
    return _process_node(state, agents, "Script Writer", "Script")

def outline_writer_node(state: AgentState, agents: Dict[str, Callable[[AgentState], str]]) -> AgentState:
    return _process_node(state, agents, "Outline Writer", "Outline")

def character_designer_node(state: AgentState, agents: Dict[str, Callable[[AgentState], str]]) -> AgentState:
    return _process_node(state, agents, "Character Designer", "Characters")

def environment_designer_node(state: AgentState, agents: Dict[str, Callable[[AgentState], str]]) -> AgentState:
    return _process_node(state, agents, "Environment Designer", "Environment")

def _process_node(state: AgentState, agents: Dict[str, Callable[[AgentState], str]], agent_role: str, part_name: str) -> AgentState:
    """Helper function to process a node and update the story part."""
    output = agents[agent_role].perform_task(state)
    state.add_story_part(part_name, output)
    return state

def setup_state_graph(agents: Dict[str, 'SimpleAgent']) -> StateGraph:
    """Setup and return the state graph with nodes and edges."""
    state_graph = StateGraph()

    state_graph.add_node("Outline Writer", outline_writer_node)
    state_graph.add_node("Character Designer", character_designer_node)
    state_graph.add_node("Environment Designer", environment_designer_node)
    state_graph.add_node("Script Writer", script_writer_node)

    state_graph.add_edge("Script Writer", "Outline Writer")
    state_graph.add_edge("Script Writer", "Character Designer")
    state_graph.add_edge("Script Writer", "Environment Designer")
    
    return state_graph
