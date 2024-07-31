import streamlit as st
import openai
import os
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define a simple agent class
class SimpleAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = PromptTemplate(template="Generate a story based on the following prompt: {prompt}")

    def generate_story(self, prompt):
        formatted_prompt = self.prompt_template.format(prompt=prompt)
        response = self.llm.Completion.create(
            engine="davinci-codex",
            prompt=formatted_prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip()

# Initialize the agent
llm = ChatOpenAI(api_key=openai_api_key)
agent = SimpleAgent(llm=llm)

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
