# AWS公式のBedrockワークショップ内にあるLangGraph用のNotebookが
# そのままだと動かなかったので単一コードにしたもの

from __future__ import annotations

import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = "us-east-1",
    runtime: Optional[bool] = True,
    external_id=None,
    ep_url=None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides"""
    target_region = region

    print(
        f"Create new client\n  Using region: {target_region}:external_id={external_id}: "
    )
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end="")
        sts = session.client("sts")
        if external_id:
            response = sts.assume_role(
                RoleArn=str(assumed_role),
                RoleSessionName="langchain-llm-1",
                ExternalId=external_id,
            )
        else:
            response = sts.assume_role(
                RoleArn=str(assumed_role),
                RoleSessionName="langchain-llm-1",
            )
        print(f"Using role: {assumed_role} ... sts::successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"][
            "SecretAccessKey"
        ]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name = "bedrock-runtime"
    else:
        service_name = "bedrock"

    if ep_url:
        bedrock_client = session.client(
            service_name=service_name,
            config=retry_config,
            endpoint_url=ep_url,
            **client_kwargs,
        )
    else:
        bedrock_client = session.client(
            service_name=service_name, config=retry_config, **client_kwargs
        )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


# ---------------------------------------------------------------------------------------

import json
import os
import sys

import boto3
import botocore

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----

# os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."

bedrock_runtime = get_bedrock_client()  #
#     assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
#     region=os.environ.get("AWS_DEFAULT_REGION", None)
# )

# ---------------------------------------------------------------------------------------

from io import StringIO
import sys
import textwrap

# from langchain.llms.bedrock import Bedrock
from typing import Optional, List, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))


# ---------------------------------------------------------------------------------------

import json

modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What is quantum mechanics? "}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "It is a branch of physics that describes how matter and energy interact with discrete energy values ",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Can you explain a bit more about discrete energies?",
            }
        ],
    },
]
body = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
    }
)

response = bedrock_runtime.invoke_model(body=body, modelId=modelId)
response_body = json.loads(response.get("body").read())
# print_ww(response_body)

# ---------------------------------------------------------------------------------------

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage

model_parameter = {"temperature": 0.0, "top_p": 0.5, "max_tokens_to_sample": 2000}
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"
react_agent_llm = ChatBedrock(
    model_id=modelId,
    client=bedrock_runtime,
    model_kwargs={"temperature": 0.1},
)

messages = [HumanMessage(content="what is the weather like in Seattle WA")]
# react_agent_llm.invoke(messages)
# print_ww(react_agent_llm.invoke(messages))

# ---------------------------------------------------------------------------------------
from langchain_aws.chat_models.bedrock import ChatBedrock

# from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# from langchain.llms.bedrock import Bedrock
# from langchain import LLMMathChain
from langchain.chains import LLMMathChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model_parameter = {"temperature": 0.0, "top_p": 0.5, "max_tokens_to_sample": 2000}
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"

modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"
chat_bedrock = ChatBedrock(
    model_id=modelId, model_kwargs={"temperature": 0.1}, client=bedrock_runtime
)

import requests

from langchain.tools import tool
from langchain.tools import StructuredTool

# from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# from langchain import LLMMathChain
from langchain.chains import LLMMathChain

headers_dict = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}


@tool("get_lat_long")
def get_lat_long(place: str) -> dict:
    """Returns the latitude and longitude for a given place name as a dict object of python."""
    url = "https://nominatim.openstreetmap.org/search"

    params = {"q": place, "format": "json", "limit": 1}
    response = requests.get(url, params=params, headers=headers_dict).json()

    if response:
        lat = response[0]["lat"]
        lon = response[0]["lon"]
        return {"latitude": lat, "longitude": lon}
    else:
        return None


@tool("get_weather")
def get_weather(latitude: str, longitude: str) -> dict:
    """Returns weather data for a given latitude and longitude."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    print_ww(f"get_weather:tool:invoked::response={response}:")
    return response.json()


# get_weather_tool = StructuredTool.from_function(get_weather)


llm_with_tools = chat_bedrock.bind_tools([get_weather, get_lat_long])
# print_ww(llm_with_tools)

# ---------------------------------------------------------------------------------------

from langchain_core.messages.human import HumanMessage

messages = [HumanMessage(content="what is the weather like in Seattle WA")]
ai_msg = llm_with_tools.invoke(messages)
# print_ww(ai_msg)

# ---------------------------------------------------------------------------------------

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.memory import ConversationBufferMemory

tools_list = [get_lat_long, get_weather]

react_agent_llm = ChatBedrock(
    model_id=modelId,
    client=bedrock_runtime,
    # model_kwargs={"max_tokens_to_sample": 100},
    model_kwargs={"temperature": 0.1},
)

# ---------------------------------------------------------------------------------------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

prompt_template_sys = """

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do, Also try to follow steps mentioned above
Action: the action to take, should be one of [ "get_lat_long", "get_weather"]
Action Input: the input to the action\nObservation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

Assistant:
{agent_scratchpad}'

"""
messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["agent_scratchpad", "input"], template=prompt_template_sys
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["input"], template="{input}")
    ),
]

chat_prompt_template = ChatPromptTemplate.from_messages(messages)
print_ww(f"from:messages:prompt:template:{chat_prompt_template}")

chat_prompt_template = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"], messages=messages
)
print_ww(f"Crafted::prompt:template:{chat_prompt_template}")

# Construct the Tools agent
react_agent = create_tool_calling_agent(
    react_agent_llm, tools_list, chat_prompt_template
)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools_list,
    verbose=True,
    max_iterations=5,
    return_intermediate_steps=True,
)
# print_ww(
#     agent_executor.invoke({"input": "Marysville WAの天気を教えてくれる？"})
# )

# ---------------------------------------------------------------------------------------

# from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# from langchain.llms.bedrock import Bedrock
# from langchain import LLMMathChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model_parameter = {"temperature": 0.0, "top_p": 0.5, "max_tokens_to_sample": 2000}
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"

# ---------------------------------------------------------------------------------------

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings

loader = PyPDFLoader("./rag_data/Amazon_SageMaker_FAQs.pdf")
bedrock_client = get_bedrock_client()
texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(
    loader.load()
)
embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)
# - create the vector store
db = FAISS.from_documents(texts, embed_model)

retriever = db.as_retriever(search_kwargs={"k": 4})
tool_search = create_retriever_tool(
    retriever=retriever,
    name="search_sagemaker_policy",
    description="Searches and returns excerpts for any question about SageMaker",
)
# print_ww(tool_search.func)
# print_ww(
#     tool_search.args_schema.schema()
# )

# ---------------------------------------------------------------------------------------

from langchain.tools.retriever import create_retriever_tool

tool_search = create_retriever_tool(
    retriever=retriever,
    name="search_sagemaker_policy",
    description="Searches and returns excerpts for any question about SageMaker",
)
# print_ww(tool_search.func)
# print_ww(
#     tool_search.args_schema.schema()
# )

# ---------------------------------------------------------------------------------------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.memory import ConversationBufferMemory

retriever_tools_list = []

retriever_tools_list.append(tool_search)

retriever_agent_llm = ChatBedrock(
    model_id=modelId,
    client=bedrock_runtime,
    # model_kwargs={"max_tokens_to_sample": 100},
    model_kwargs={"temperature": 0.1},
)

prompt_template_sys = """

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do, Also try to follow steps mentioned above
Action: the action to take, should be one of [ "get_lat_long", "get_weather"]
Action Input: the input to the action\nObservation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

Assistant:
{agent_scratchpad}'

"""
messages = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["agent_scratchpad", "input"], template=prompt_template_sys
        )
    ),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["input"], template="{input}")
    ),
]

chat_prompt_template = ChatPromptTemplate.from_messages(messages)
print_ww(f"from:messages:prompt:template:{chat_prompt_template}")

chat_prompt_template = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"], messages=messages
)
# print_ww(f"Crafted::prompt:template:{chat_prompt_template}")

# react_agent_llm.bind_tools = custom_bind_func

# Construct the Tools agent
retriever_agent = create_tool_calling_agent(
    retriever_agent_llm, retriever_tools_list, chat_prompt_template
)
agent_executor_retriever = AgentExecutor(
    agent=retriever_agent,
    tools=retriever_tools_list,
    verbose=True,
    max_iterations=5,
    return_intermediate_steps=True,
)
# agent_executor_retriever.invoke({"input": "Amazon SageMaker Clarifyとは何ですか？"})

# ---------------------------------------------------------------------------------------

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.memory import ConversationBufferMemory

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Type,
    Union,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
    convert_to_openai_function,
)
from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

model_parameter = {"temperature": 0.0, "top_p": 0.5, "max_tokens_to_sample": 2000}
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-v2"

# ---------------------------------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.llms import Bedrock
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

# ["weather", "search_sagemaker_policy" ] #-"SageMaker"]

from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms.bedrock import Bedrock
from langchain import LLMMathChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

model_parameter = {"temperature": 0.0, "top_p": 0.5, "max_tokens_to_sample": 2000}
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

members = ["weather_search", tool_search.name]
print(members)
options = ["FINISH"] + members

print(options)
prompt_finish_template_simple = """
Given the conversation below who should act next?
Current Conversation: {history_chat}

Or should we FINISH? ONLY return one of these {options}. Do not explain the process.Select one of: {options}

Question: {input}
"""

supervisor_llm = ChatBedrock(
    model_id=modelId,
    client=bedrock_runtime,
)

simple_supervisor_chain = (
    # {"input": RunnablePassthrough()}
    RunnablePassthrough()
    | ChatPromptTemplate.from_template(prompt_finish_template_simple)
    | supervisor_llm
    | ToolsAgentOutputParser()  # StrOutputParser()
)

simple_supervisor_chain.invoke(
    {"input": "SageMakerって何？", "options": options, "history_chat": ""}
)

# ---------------------------------------------------------------------------------------

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# The agent state is the input to each node in the graph
class GraphState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next_node' field indicates where to route to next
    next_node: str
    # - initial user query
    user_query: str
    # - # instantiate memory
    convo_memory: ConversationBufferMemory

    options: list


def input_first(state: GraphState) -> Dict[str, str]:
    print_ww(f"""start input_first()....::state={state}::""")
    init_input = state.get("user_query", "").strip()

    # store the input and output
    # - # instantiate memory since this is the first node
    convo_memory = ConversationBufferMemory(
        human_prefix="\nHuman", ai_prefix="\nAssistant", return_messages=False
    )  # - get it as a string
    convo_memory.chat_memory.add_user_message(init_input)
    # convo_memory.chat_memory.add_ai_message(ai_output.strip())

    options = ["FINISH", "weather_search", tool_search.name]

    # return {"messages": [SystemMessage(content="This is a system message"),HumanMessage(content=init_input, name="user_input")]}
    return {"user_query": init_input, "options": options, "convo_memory": convo_memory}


def agent_node(state, agent_return, name):
    result = {"output": f"hardcoded::Agent:name={name}::"}  # agent.invoke(state)
    # - agent.invoke(state)

    init_input = state.get("user_query", "").strip()
    state.get("convo_memory").chat_memory.add_user_message(init_input)
    state.get("convo_memory").chat_memory.add_ai_message(
        agent_return
    )  # f"SageMaker clarify helps to detect bias in our ml programs. There is no further information needed.")#result.return_values["output"])

    return {"next_node": END}


def retriever_node(state: GraphState) -> Dict[str, str]:
    print_ww(
        f"\nuse this to go the retriever way to answer the question():: state::{state}"
    )
    # agent_return = retriever_agent.invoke()

    init_input = state.get("user_query", "").strip()
    agent_return = agent_executor_retriever.invoke({"input": init_input})["output"][
        :-100
    ]
    # agent_return = "SageMaker clarify helps to detect bias in our ml programs. There is no further information needed."
    return agent_node(state, agent_return, tool_search.name)


def weather_node(state: GraphState) -> Dict[str, str]:
    print_ww(f"\nuse this to answer about the weather state::{state}::")
    # agent_return = react_agent.invoke()
    init_input = state.get("user_query", "").strip()
    agent_return = agent_executor.invoke({"input": init_input})["output"][:-100]
    # agent_return = "Weather is nice and bright and sunny with temp of 54 and winds from North at 2 miles per hour. Nothing more to report"
    return agent_node(state, agent_return, name="weather_search")


def error(state: GraphState) -> Dict[str, str]:
    print_ww(f"""start error()::state={state}::""")
    return {"final_result": "error", "first_word": "error", "second_word": "error"}


def supervisor_node(state: GraphState) -> Dict[str, str]:
    print_ww(f"""supervisor_node()::state={state}::""")  # agent.invoke(state)
    # -
    init_input = state.get("user_query", "").strip()
    # messages = state.get("messages", [])
    options = state.get("options", ["FINISH", "weather_search", tool_search.name])
    # print_ww(f"supervisor_node()::options={options}::")
    convo_memory = state.get("convo_memory")
    history_chat = convo_memory.load_memory_variables({})["history"]
    print(f"supervisor_node():History of messages so far :::{history_chat}")
    # - AgentFinish(return_values={'output': 'Search_sagemaker_policy'}, log='Search_sagemaker_policy')
    # result = supervisor_chain.invoke({"input": init_input, "messages": messages, "intermediate_steps": []}) # - does not work due to chat template
    # supervisor_chain.invoke({"input": "What is sagemaker", "messages": [], "intermediate_steps": []}) #- works is complicated

    result = simple_supervisor_chain.invoke(
        {"input": init_input, "options": options, "history_chat": history_chat}
    )
    print_ww(f"supervisor_node():result={result}......")

    # state.get("convo_memory").chat_memory.add_user_message(init_input)
    convo_memory.chat_memory.add_ai_message(result.return_values["output"])

    return {"next_node": result.return_values["output"]}


workflow = StateGraph(GraphState)
workflow.add_node(tool_search.name, retriever_node)
workflow.add_node("weather_search", weather_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("init_input", input_first)
# print(workflow)

# ---------------------------------------------------------------------------------------

# - #["weather", "search_sagemaker_policy" ] #-"SageMaker"]
members = ["weather_search", tool_search.name, "init_input"]

print_ww(f"members of the nodes={members}")

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next_node"], conditional_map)

# - add end just for the WEATHER --
workflow.add_edge("weather_search", END)

# Finally, add entrypoint
workflow.set_entry_point("init_input")  # - supervisor")

graph = workflow.compile()
# print(graph)

# graph.get_graph().print_ascii()

# ---------------------------------------------------------------------------------------

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

# [SystemMessage(content="This is a system message"), HumanMessage(content="Amazon SageMaker Clarifyって何？")]

graph.invoke(
    {
        "user_query": "Amazon SageMaker Clarifyって何？日本語で答えて",
        "recursion_limit": 1,
    }
)
