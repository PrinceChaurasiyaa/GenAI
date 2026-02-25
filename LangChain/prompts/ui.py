from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import load_prompt

llm = ChatOllama(
  model="llama3"
)

st.header('Research Tool')

prompts_template = load_prompt('template.json')



paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-ShotLearners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "CodeOriented", "Mathematical"] )
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5paragraphs)", "Long (detailed explanation)"] )





if st.button('Summarize'):
  chain = prompts_template | llm
  response = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
  })

  
  st.write(response.content)