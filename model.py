from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


from dotenv import load_dotenv
import os
import langsmith
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TextSplitter
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


embedding = OpenAIEmbeddings( )

vectorstore = FAISS.from_documents(docs, embedding)

retriever = vectorstore.as_retriever()


from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
# Step 5: Initialize the OpenAI language model
llm = ChatOpenAI(temperature=0.7,model='gpt-4-turbo')


# Step 6: Create the RetrievalQA Chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Step 7: Query the chain
query = "How is the construction of hydrogen pipelines referenced in the activity “Transmission and distribution networks for renewable and low-carbon gases” applied"
response = rag_chain.run(query)
print(response)


#
# chunks = split_text(large_text, chunk_size=1500)  # Chunk size is adjusted to fit within the token limit
#
# # Process each chunk individually
# responses = []
# for chunk in chunks:
#     response = llm(chunk)
#     responses.append(response)
#
# # Combine responses if necessary
# final_response = " ".join(responses)
# print(final_response)
#
#




















# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content
#
# pipeline("What is the capital of Belgium")





