# CapstructAI
## Building Code and Construction Safety Assistant

Repo for a Retrieval Augmented Generation (RAG) app that references construction safety and building codes relevant to Vancouver, British Columbia, Canada.  

[Link](https://capstructai.streamlit.app) to app hosted on Streamlit community cloud. 
  
[Submission](https://devpost.com/software/capstructai-simplifying-construction-code-compliance-with-ai) for the Rag N Roll hackathon hosted by Snowflake.

[Project demo and presentation](https://youtu.be/4VeNttm9grU)

### Contributors:  
Kenneth Fiawoyife - Product Owner, Product Manager  
Vincent Ji - Lead Developer  

### Repo overview:

`streamlit_app.py` and `requirements.txt` are the files referenced by the Streamlit Community Cloud to run the app. 

`sql_worksheet.txt` contains the SQL queries used to set up the databases and Cortex search service on Snowflake

`trulens_eval.py` contains the Python code which uses Trulens to evaluate app performance, with and without Trulens integration
