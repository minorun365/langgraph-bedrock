import os
import json
import requests
from typing import Dict, List, TypedDict

import boto3
from botocore.config import Config
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

# Bedrock client setup
def get_bedrock_client(region="us-east-1"):
    session = boto3.Session()
    return session.client(
        service_name="bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )

bedrock_runtime = get_bedrock_client()

# ChatBedrock model setup
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
chat_bedrock = ChatBedrock(
    model_id=modelId,
    client=bedrock_runtime,
    model_kwargs={"temperature": 0.1}
)

# Tool definitions
def get_lat_long(place: str) -> dict:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers).json()
    if response:
        return {"latitude": response[0]["lat"], "longitude": response[0]["lon"]}
    return None

def get_weather(latitude: str, longitude: str) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    return response.json()

# SageMaker FAQ retriever setup
loader = PyPDFLoader("./rag_data/Amazon_SageMaker_FAQs.pdf")
texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(loader.load())
embed_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)
db = FAISS.from_documents(texts, embed_model)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Tools
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Get weather data for a given latitude and longitude."
)

lat_long_tool = Tool(
    name="get_lat_long",
    func=get_lat_long,
    description="Get latitude and longitude for a given place name."
)

retriever_tool = Tool(
    name="search_sagemaker_policy",
    func=retriever.get_relevant_documents,
    description="Searches and returns excerpts for any question about SageMaker"
)

tools = [weather_tool, lat_long_tool, retriever_tool]

# Agent setup
prompt_template = """
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

{agent_scratchpad}
"""

chat_prompt = ChatPromptTemplate.from_template(prompt_template)

# Create tool_names string
tool_names = ", ".join([tool.name for tool in tools])

agent = create_tool_calling_agent(chat_bedrock, tools, chat_prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# Graph state and nodes
class GraphState(TypedDict):
    messages: List[str]
    next_node: str
    user_query: str
    memory: ConversationBufferMemory

def input_node(state: GraphState) -> Dict:
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_user_message(state["user_query"])
    return {"memory": memory, "next_node": "agent"}

def agent_node(state: GraphState) -> Dict:
    result = agent_executor.invoke({
        "input": state["user_query"],
        "tool_names": tool_names
    })
    state["memory"].chat_memory.add_ai_message(result["output"])
    return {"next_node": END}

# Graph setup
workflow = StateGraph(GraphState)
workflow.add_node("input", input_node)
workflow.add_node("agent", agent_node)
workflow.add_edge("input", "agent")
workflow.set_entry_point("input")

graph = workflow.compile()

# Invoke the graph
result = graph.invoke({
    "user_query": "明日の東京の天気は？ あと、Amazon SageMaker Autopilot と Automatic Model Tuningってどう使い分ければいいの？ いずれも日本語で答えて。",
    "recursion_limit": 2,
})