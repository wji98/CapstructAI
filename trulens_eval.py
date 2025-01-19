from snowflake.snowpark import Session
from trulens.core import Feedback, Select, TruSession
import numpy as np
from trulens.apps.custom import TruCustomApp
from snowflake.core import Root
from snowflake.cortex import Complete
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.providers.cortex.provider import Cortex
from trulens.apps.custom import instrument

import json
from snowflake.snowpark.context import get_active_session
from trulens.core.guardrails.base import context_filter
import nltk

# Download the NLTK data
nltk.download('punkt_tab')

NUM_CHUNKS = 5
MIN_SCORE = 0.6

CORTEX_SEARCH_DATABASE = "CC_QUICKSTART_CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"

class CapstructAI:

    def __init__(self ,svc):
        self.svc = svc
        self.chat_history = []
        self.columns = [
                        "chunk",
                        "relative_path",
                        "category"
                        ]
    
    @instrument        
    def retrieve_context(self, query):
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
        
        if cat == "ALL":
            response = self.svc.search(query, self.columns, limit=NUM_CHUNKS)
        else:
            filter_obj = {"@eq":{"category": cat}}
            response = self.svc.search(query, self.columns, filter=filter_obj, limit=NUM_CHUNKS)
                
        if response.results:
            results =  [curr["chunk"] for curr in response.results]
            ret = response.json()
        else:
            results = []
            ret = ""
        
        return results, ret
        
    def optimize_query(self, question):
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
            {self.chat_history}
            </chat_history>
            """
        
        sumary = Complete('mistral-large2', prompt)   
    
        sumary = sumary.replace("'", "")
    
        return sumary
    
    def create_prompt(self, myquestion):
    
        optimized_query = self.optimize_query(myquestion)
        _, prompt_context = self.retrieve_context(optimized_query)
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
               {self.chat_history}
               </chat_history>
               <context>          
               {prompt_context}
               </context>
               <question>  
                {optimized_query}
               </question>
               Answer: 
               """
    
        return prompt

    @instrument
    def query(self, myquestion):
    
        prompt = self.create_prompt(myquestion)
    
        response = Complete('mistral-large2', prompt)  

        self.chat_history.append({"role": "user", "content": myquestion})
        self.chat_history.append({"role": "assistant", "content": response})
        return response
    
def main():
    connection_params = {
      "account":  "<account>",
      "user": "<user>",
      "password": "<password>",
      "role": "<role>",
      "database": CORTEX_SEARCH_DATABASE,
      "schema": CORTEX_SEARCH_SCHEMA,
      "warehouse": "COMPUTE_WH"
    }
    
    snowpark_session =  Session.builder.configs(connection_params).create()
    root = Root(snowpark_session)                         
    svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]
    
    snowpark_connector = SnowflakeConnector(snowpark_session=snowpark_session)
    tru_session = TruSession(connector=snowpark_connector)
    
    provider = Cortex(snowpark_session=snowpark_session, model_engine="llama3.1-8b")

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(Select.RecordCalls.retrieve_context.rets[0][:].collect())
        .on_output()
    )
    
    f_context_relevance = (
        Feedback(provider.context_relevance, name="Context Relevance")
        .on_input()
        .on(Select.RecordCalls.retrieve_context.rets[0][:])
        .aggregate(np.mean)
    )
    
    f_answer_relevance = (
        Feedback(provider.relevance, name="Answer Relevance")
        .on_input()
        .on_output()
        .aggregate(np.mean)
    )
    
    rag = CapstructAI(svc)
    tru_rag = TruCustomApp(
        rag,
        app_name="CapstructAI",
        app_version="simple",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
        )
        
    prompts = ["What are the structural integrity requirements for foundation systems in Vancouver for buildings over 100 feet tall, particularly in seismic zones?",
               "What are the fire protection and smoke ventilation requirements for underground parking garages in Vancouver according to the BC Building Code?",
               "What are the specific design requirements for load-bearing walls in multi-story commercial buildings under Vancouver’s seismic regulations?",
               "What are the ventilation system requirements for industrial facilities in Vancouver that handle hazardous materials to ensure worker safety?",
               "What are the energy efficiency and insulation requirements for residential buildings in Vancouver, particularly in terms of thermal resistance (R-values)?",
               "What are the construction site signage requirements for hazardous areas, such as those with high-voltage equipment, in Vancouver?",
               "What are the requirements for soil contamination testing before commencing construction in Vancouver?"
               ]
    
    with tru_rag as recording:
        for prompt in prompts:
            print(prompt)
            response = rag.query(prompt)
            print(response)
    print("getting results")
    tru_session.get_leaderboard()

    f_context_relevance_score = (
        Feedback(provider.context_relevance, name="Context Relevance")
    )

    class CapstructAI_v1(CapstructAI):
        
        @instrument
        @context_filter(f_context_relevance_score, MIN_SCORE, keyword_for_prompt="query")
        def retrieve_context(self, query):
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
            
            if cat == "ALL":
                response = self.svc.search(query, self.columns, limit=NUM_CHUNKS)
            else:
                filter_obj = {"@eq":{"category": cat}}
                response = self.svc.search(query, self.columns, filter=filter_obj, limit=NUM_CHUNKS)
                        
            if response.results:
                results =  [curr["chunk"] for curr in response.results]
                ret = response.json()
            else:
                results = []
                ret = ""
                
            return results, ret

    improved_rag = CapstructAI_v1(svc)
    tru_filtered_rag = TruCustomApp(
        improved_rag,
        app_name="CapstructAI",
        app_version="improved",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )

    with tru_filtered_rag as recording:
        for prompt in prompts:
            print(prompt)
            response = improved_rag.query(prompt)
            print(response)
    
    tru_session.get_leaderboard()

if __name__=='__main__':
    main()
