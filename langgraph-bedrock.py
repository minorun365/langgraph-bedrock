# Pythonの外部ライブラリをインポート
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrockConverse  # BedrockのConverse APIを利用
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


# 「検索」ツールを定義（中身は仮のもの、テキストを返すだけ）
@tool
def search(query: str):
    """Call to surf the web."""
    if "東京" in query:
        return "東京は今日も最高気温35度越えの猛暑です。"
    return "日本は今日、全国的に晴れです。"


# 使えるツール一覧を定義
tools = [search]

# 「ツールを実行する」関数（風のインスタンス）を定義
tool_node = ToolNode(tools)

# LLMとTool useを定義（ローカルへAWSのIAMアクセスキーを設定しておいてください）
model = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0"
).bind_tools(tools)


# 「次に進む」という関数を定義
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # LLMが「ツールを使うべし」と判断したら、「tools」ノードに進む
    if last_message.tool_calls:
        return "tools"
    # そうでなければワークフローを終了する
    return END


# 「LLMを呼ぶ」という関数を定義
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# グラフを定義
workflow = StateGraph(MessagesState)

# グラフにノードを2つ追加（エージェントとツール）
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# グラフの開始地点を「エージェント」ノードに設定
workflow.set_entry_point("agent")

# グラフに「条件付きエッジ」を追加（エージェントが呼ばれた後は「次へ進む」を実行）
workflow.add_conditional_edges("agent", should_continue)

# グラフに「エッジ」を追加（ツール → エージェントの経路）
workflow.add_edge("tools", "agent")

# ステート保持用のメモリを初期化
checkpointer = MemorySaver()

# ここまで作ったワークフローをコンパイルして、LangChainのRunnableで呼び出せるようにする
app = workflow.compile(checkpointer=checkpointer)

# グラフの構造をターミナルに表示
app.get_graph().print_ascii()

# ワークフローを実行
final_state = app.invoke(
    {"messages": [HumanMessage(content="東京の天気は？")]},
    config={"configurable": {"thread_id": 42}},
)
result = final_state["messages"][-1].content

# 結果をターミナルに表示
print(result)
