import streamlit as st # Import python packages
#from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from snowflake.cortex import Complete
from snowflake.core import Root
from snowflake.connector import connect
from snowflake.snowpark.context import get_active_session

import pandas as pd
import json

pd.set_option("max_colwidth",None)

### Default Values
NUM_CHUNKS = 5 # Num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 7 # how many last conversations to remember. This is the slide window.

# service parameters
CORTEX_SEARCH_DATABASE = "CC_QUICKSTART_CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
######
######

# columns to query in the service
COLUMNS = [
    "chunk",
    "relative_path",
    "category"
]

connection_params = {
      "account": st.secrets["ACCOUNT"],
      "user": st.secrets["USER"],
      "password": st.secrets["PASSWORD"],
      "role": st.secrets["ROLE"],
      "database": CORTEX_SEARCH_DATABASE,
      "schema": CORTEX_SEARCH_SCHEMA,
      "warehouse": "COMPUTE_WH"
    }
#session =  Session.builder.configs(connection_params).create()
#root = Root(session)                         
#svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
#connection = connect(**connection_params)
#root = Root(connection)
#session = None
#@st.cache_resource
#def get_cortex_service():
#    session =  Session.builder.configs(connection_params).create()
#    root = Root(session)
#    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
#    return svc

#svc = get_cortex_service()
#session = get_active_session()

st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None) 

debug = False

def initialize_snowpark_session():
    global session
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        session =  Session.builder.configs(connection_params).create()
        st.session_state.initialized = True
    else:
        session = get_active_session()

def config_options():

    #categories = session.table('docs_chunks_table').select('category').distinct().collect()
    
    #cat_list = []
    #for cat in categories:
    #    cat_list.append(cat.CATEGORY)
    
    st.sidebar.button("Start New Conversation (Clears Chat History)", key="clear_conversation", on_click=init_messages)
    #if debug:
    #    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():

    # Initialize chat history
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks_search_service(query):
    #session =  Session.builder.configs(connection_params).create()
    #root = Root(session)                         
    #svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
    
    prompt = f"""
        Based on the QUESTION in between the <question> and </question> tags, if the user explicitly asks to search for a specific 
        category of documents that matches one of the categories below, then answer in one word from the options below. Simply having the 
        word in the question is not sufficient, the user must ask for an answer using the category of documents:
        1. Safety
        2. Building Code
        3. Sustainability
        4. Plumbing
        5. Fire
        6. Electrical

        In all other cases, answer "ALL"
        
        <question>
        {query}
        </question>
        """
    cat = Complete('mistral-large2', prompt)
    cat = cat.replace("'", "").strip()
    #if st.session_state.category_value == "All Building and Safety Codes":

    st.sidebar.text("Category")
    st.sidebar.caption(cat)

    if cat == "ALL":
        response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)
    else:
        filter_obj = {"@eq":{"category": cat}}
        response = svc.search(query, COLUMNS, filter=filter_obj, limit=NUM_CHUNKS)

    if debug:
        st.sidebar.json(response.json())
    
    return response.json()  

def get_chat_history():
#Get the history from the st.session_stage.messages according to the slide window parameter
    
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history

def summarize_question_with_history(chat_history, question, svc):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        Based on the CHAT HISTORY between the <chat_history> and </chat_history> tags and the QUESTION between the <question> and </question> tags, 
        generate a query that extends the QUESTION with the CHAT HISTORY provided. 
        After you are done generating the query, optimize it so that it is easier for your to understand, whether it is by writing out commonly used abbreviations, or narrowing ambiguities. 
        The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    
    sumary = Complete('mistral-large2', prompt)   

    sumary = sumary.replace("'", "")

    return sumary

def optimize_query(chat_history, question):
    prompt = f"""
        Based on the QUESTION between the <question> and </question> tags, 
        generate a query that is easier for you to understand, whether it is by writing out commonly used abbreviations, or narrowing ambiguities. 
        If the user attempts to rate your response in the QUESTION, generate a prompt commanding you to thank the user for their feedback if it is positive, or 
        apologize and promise to do better if it is negative.
        If the query is explicitly referencing previous information given by either you or the user, extend the QUESTION with the CHAT HISTORY 
        provided between the <chat_history> and </chat_history> tags.
        The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <question>
        {question}
        </question>
        <chat_history>
        {chat_history}
        </chat_history>
        """
    
    sumary = Complete('mistral-large2', prompt)   

    sumary = sumary.replace("'", "")

    return sumary
    
    
def create_prompt (myquestion, svc):

    chat_history = get_chat_history()
    optimized_query = optimize_query(chat_history, myquestion)
    prompt_context = get_similar_chunks_search_service(optimized_query, svc)
    #chat_history = ""
    
    st.sidebar.text("Optimized query:")
    st.sidebar.caption(optimized_query)
    prompt = f"""
           You are an expert chat assistant that extracts information from the CONTEXT provided
           between <context> and </context> tags.
           You offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags.
           When answering the question contained between <question> and </question> tags
           be concise and do not hallucinate.  
           If you don't have the information, just say so.
           
           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your answer.

           If you can't answer the question from the CONTEXT provided, answer ignoring the CONTEXT and CHAT HISTORY, but also state that 
           "This is the best answer I can provide with the available data. Please verify the information with the reference documents linked in 
           the sidebar to ensure full compliance with relevant regulations." and bold the text if possible.
           
           If the user attempts to rate your response, either thank the user for their feedback if it is positive, or apologize and promise to do better
           if it is negative.
           
           <chat_history>
           {chat_history}
           </chat_history>
           <context>          
           {prompt_context}
           </context>
           <question>  
           {optimized_query}
           </question>
           Answer: 
           """
    
    json_data = json.loads(prompt_context)

    relative_paths = set(item['relative_path'] for item in json_data['results'])

    return prompt, relative_paths

def answer_question(myquestion):

    session = Session.builder.configs(connection_params).create()
    root = Root(session)
    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
    
    prompt, relative_paths =create_prompt (myquestion, svc)
    response = Complete('mistral-large2', prompt)
    
    session.close()
    return response, relative_paths

def export_chat_history():
    ret = ""
    for message in st.session_state.messages:
        ret += message["role"]
        ret += ": "
        ret += message["content"]
        ret += "\n"
    return ret

def delete_conversation():
    st.session_state.messages = []
    
def main():
    
    #st.title(f":speech_balloon: Chat with StructAI - Your Building Code and Construction Safety Assistant")
    #with col3:
    
    st.title("CapstructAI")
    st.subheader("Your Building Code and Construction Safety Assistant")

    #initialize_snowpark_session()
    config_options()
    init_messages()

    #connection = connect(**connection_params)
    
    with st.expander(label='Chat History'):
        with st.container(height=500):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                st.download_button("Download Chat", data=export_chat_history(), file_name="chat_history.txt")
            with col3:
                st.button("Clear Chat", key="delete_convo", on_click=delete_conversation)

    st.sidebar.subheader("Relevant Documents:")
    
    input_field = st.container()
    with input_field:
        col1, col2 = st.columns([6,1])
        with col1:
            question = st.text_input("Enter question", placeholder="Ask me a question", label_visibility="collapsed")
        with col2:
            send_button = st.button(":arrow_forward:")
    
    #if question := st.text_input("Enter question", placeholder="Ask me a question", label_visibility="collapsed") or st.button("Send"):
    if question or send_button:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
    
            question = question.replace("'","")
    
            with st.spinner("Thinking..."):
                response, relative_paths = answer_question(question)            
                response = response.replace("'", "")
                message_placeholder.markdown(response)

                if relative_paths != "None":
                    with st.sidebar.expander("Related Documents"):
                        for path in relative_paths:
                            cmd2 = f"select GET_PRESIGNED_URL(@docs, '{path}', 360) as URL_LINK from directory(@docs)"
                            df_url_link = session.sql(cmd2).to_pandas()
                            url_link = df_url_link._get_value(0,'URL_LINK')
                
                            display_url = f"Doc: [{path}]({url_link})"
                            st.sidebar.markdown(display_url)

        
        st.session_state.messages.append({"role": "assistant", "content": response})
    #session.close()
    # Display chat messages from history on app rerun
        
if __name__ == "__main__":
    main()
    
