from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


os.environ['LANGCHAIN_PROJECT']='Sequential LLM'



load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)
model2 = ChatGroq(model="llama-3.1-8b-instant",temperature=0.5)

parser = StrOutputParser()


chain = prompt1 | model1 | parser | prompt2 | model2 | parser


config:dict={
    "run_name":"Seq Chain",
    'tags' : ["LLM app","report gen","summ."],
    'metadata' : {"model1":"ChatGroq","model2":'chatGroq',"Parser":"StrOut"}
}

result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
