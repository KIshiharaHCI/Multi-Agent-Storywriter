from typing import Dict
from state import StateGraph, AgentState, setup_state_graph
from agents import SimpleAgent, setup_agents
from config import load_and_configure_environment


class Director:
    def __init__(self, state_graph: StateGraph, agents: Dict[str, SimpleAgent]):
        self.state_graph = state_graph
        self.agents = agents
        self.current_node = "Script Writer"

    def run_workflow(self, initial_prompt: str) -> str:
        """Run the workflow from the initial prompt."""
        state = AgentState(
            messages=[{"role": "user", "content": initial_prompt}],
            next_agent=self.current_node,
            story_parts={}
        )
        
        while state["next_agent"]:
            current_func = self.state_graph.nodes[state["next_agent"]]
            state = current_func(state, self.agents)
            print(f"Output from {state['next_agent']}: {state.get_story_content()}")  # debugging
            
            next_nodes = self.state_graph.get_next_node(state["next_agent"])
            state["next_agent"] = next_nodes.pop(0) if next_nodes else None
        
        return self.review_story(state)

    def review_story(self, state: AgentState) -> str:
        """Review the final story content."""
        story_content = state.get_story_content()
        review_prompt = "Review the following story for coherence and completeness, ensuring it reads as a \
            cohesive narrative. Focus only on the story content itself. Remove any outlines, character des-\
            criptions, or other extraneous parts. Remove all parts including the Outline that are not ne-  \
            cessary and only show the story. Structure as follows: a title and the main body of the story, \
            formatted to ensure readability and coherence. Omit any introductory or concluding remarks, as \
            well as anything else other than the title and main. Ensure there is no indication of AI invol-\
            vement. Make any necessary adjustments:\n\n" + story_content 
        review_state = AgentState(
            messages=[{"role": "user", "content": review_prompt}],
            next_agent="Script Writer",
            story_parts={}
        )
        return self.agents["Script Writer"].perform_task(review_state)

def generate_story_from_prompt(prompt: str) -> str:
    """Generate a story from the prompt."""
    load_and_configure_environment()
    agents = setup_agents()
    state_graph = setup_state_graph(agents)
    director = Director(state_graph, agents)
    return director.run_workflow(prompt)
