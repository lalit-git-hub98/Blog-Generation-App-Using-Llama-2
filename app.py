import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from huggingface_hub import login

to1 = "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"

login(to1)

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens':300,
                              'temperature':0.5})
    
    ## Prompt Template

    template="Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    return response


st.set_page_config(page_title="Blog Generation Using LLama 2",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Blog Generation App :earth_americas:")

st.write('Although the app uses quantized version of llama-2, it might take some time to generate the respone!')

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers', 'Engineers', 'Doctors', 'Content Creators', 'Sportsman', 'Businessman', 'Common People'),index=0)
    
submit=st.button("Generate Blog")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))