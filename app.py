import streamlit as st
from director import generate_story_from_prompt

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
                story = generate_story_from_prompt(prompt)
                st.subheader("Generated Story")
                st.markdown(f'<div class="story-output">{story}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a prompt.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
