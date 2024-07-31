import streamlit as st
import openai
import os
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()
# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

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
        result = self.executor.invoke({"messages": messages})
        return result["messages"][-1]["content"].strip()


# Initialize the agent
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
executor = create_agent(llm)
agent = SimpleAgent(executor=executor)


def main():
    st.set_page_config(page_title="Collaborative Story Writing with AI", layout="centered")
    st.title("Collaborative Story Writing with AI")
    
    st.markdown("""
        <style>
        .main {background-color: #f0f0f0; padding: 20px; border-radius: 10px;}
        .prompt-input {margin-top: 20px; margin-bottom: 20px;}
        .submit-button {margin-bottom: 20px;}
        .story-output {margin-top: 20px;}
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    prompt = st.text_area("Enter your story prompt:", "", key="prompt-input")
    if st.button("Generate Story", key="submit-button"):
        if prompt:
            with st.spinner('Generating story...'):
                story = agent.generate_story(prompt)
                st.subheader("Generated Story")
                st.markdown(f'<div class="story-output">{story}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a prompt.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
