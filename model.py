from dotenv import load_dotenv
import os
import openai
import time
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load a specific .env file (e.g., for development)
load_dotenv(dotenv_path='keys.env')
# Access environment variables
api_key = os.getenv('LANGCHAIN_API_KEY')
openai_api_key=os.getenv('OPENAI_API_KEY')
# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())
loader = TextLoader("./taxonomy_faqs_cleaned.md", encoding='utf-8')
data=loader.load()
text_splitter = CharacterTextSplitter(separator= '###')
docs = text_splitter.split_documents(data) #Splitted Text is saved in docs
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0.3,model='gpt-4-turbo')
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


### RAG Module
query="What is Ammonia considered as?"
response = rag_chain.invoke(query)
print(response['result'])




### LLMOps Module
@traceable
def pipeline(model,query,vectorstore,temperature):
    embedding = OpenAIEmbeddings()
    if vectorstore=='faiss':
        vectorstore = FAISS.from_documents(docs, embedding)
    if vectorstore=='chroma':
        vectorstore = Chroma.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=temperature,model=model)


    # Step 6: Create the RetrievalQA Chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Step 7: Query the chain
    start_time = time.time()
    response = rag_chain.invoke(query)
    end_time = time.time()
    response_time = end_time - start_time
    #print(response['result'])
    return{'response':response['result'],'response_time':response_time}


expected_answer='Article 19(5) of the Taxonomy Regulation requires the Commission to regularly review the TSC defining substantial contribution to environmental objectives and DNSH to those objectives. In case of activities identified as transitional in the Climate Delegated Act, the review would be conducted at least every three years to ensure the criteria remain on a credible transition pathway consistent with a climate-neutral economy. No minimum period is specified for the other activities. The TSC will be updated over time to keep them aligned with overall policy objectives, technological developments and the availability of scientifically robust evidence justifying the introduction of new or updated criteria.'
def answer_evaluator(model,temperature,answer, input_question,output_answer):
    grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")
    grade_prompt_hallucinations = prompt = hub.pull("langchain-ai/rag-answer-hallucination")
    # Get question, ground truth answer, RAG chain answer
    input_question = input_question
    reference = output_answer
    prediction = answer
    # LLM grader
    llm = ChatOpenAI(model=model, temperature=temperature)
    # Structured prompt
    answer_grader_accuracy = grade_prompt_answer_accuracy | llm
    answer_grader_hallucination = grade_prompt_hallucinations | llm

    # Run evaluator
    score_accuracy = answer_grader_accuracy.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score_accuracy = score_accuracy["Score"]

    score_hallucination = answer_grader_hallucination.invoke({"documents": reference,
                                                    "student_answer": prediction})
    score_hallucination = score_hallucination["Score"]
    return {'answer_v_reference_score': score_accuracy,'hallucination':score_hallucination}


today = datetime.today()
temperature_list=[0,0.3,0.7]
model_list=['gpt-3.5-turbo','gpt-4-turbo']

#Check if today is Sunday
if today.weekday() == 0:
    print('Model will be finetuned today')
    curr_time=100
    chosen_model=''
    chosen_temperature=0
    score=0
    for temperature in temperature_list:
        for model in model_list:
            output=pipeline(model=model, query=query,vectorstore='faiss', temperature=temperature)
            prev_score=score
            score=answer_evaluator(model,temperature,expected_answer,query,output['response'])['answer_v_reference_score']
            if output['response_time'] < curr_time and score <= prev_score :
                curr_time=output['response_time']
                chosen_temperature=temperature
                chosen_model=model
    os.environ['model'] = chosen_model
    os.environ['temperature'] = str(chosen_temperature)

    file_path = 'hyperparameters.env'
    env_vars = {
        'model': chosen_model,
        'temperature': chosen_temperature
    }
    # Open the file for writing the chosen fine-tuned hyper-parameters
    with open(file_path, 'w') as env_file:
        for key, value in env_vars.items():
            env_file.write(f"{key}={value}\n")


load_dotenv(dotenv_path='hyperparameters.env')

# Access environment variables
model = os.getenv('model')
temperature=os.getenv('temperature')

pipeline(model,query,vectorstore,temperature)



