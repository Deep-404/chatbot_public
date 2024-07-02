import os
import openai
import streamlit as st

from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader


with st.sidebar:
    st.title("'🤗💬 Chat with your Data'")
    st.markdown('''
  
                ''')
    

def main():
    st.header("Chat with your Data")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=False)
    docs = reader.load_data()

    apikey=st.text_input("whats the api key: ")
    if apikey:
        os.environ["OPENAI_API_KEY"] = apikey
    
    openai.api_key = os.environ["OPENAI_API_KEY"]

    loc_llm = OpenAI(model="gpt-3.5-turbo")

    service_context = ServiceContext.from_defaults(llm=loc_llm)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query=st.text_input("Ask questions related to your Data")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)

if __name__=='__main__':
    main()  