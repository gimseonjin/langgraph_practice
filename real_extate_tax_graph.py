# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing import TypedDict


class AgentState(TypedDict):
    question: str
    answer: str
    tax_base_equation: str # 과세표준 계산 수식
    tax_deduction: str # 공제액
    market_ratio: str # 공정시장가액비율
    tax_base: str # 과세표준 계산

# %%
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

# %%
# vector store 생성
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="real_estate_tax",
    persist_directory="./real_estate_tax_collection",
    embedding_function=embeddings
)

retriver = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
question = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'

# %%
# llm 설정
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o")
mini_llm = ChatOpenAI(model="gpt-4o-mini")

# %%

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요.  부연 설명 없이 수식만 리턴해주세요"),
    ("human", "{tax_base_equation}")
])

tax_base_retrieval_chain = (
    {"context": retriver, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

tax_base_equation_chain = (
    {"tax_base_equation": RunnablePassthrough()} 
    | tax_base_equation_prompt 
    | llm 
    | StrOutputParser()
)

tax_base_chain = { "tax_base_equation" : tax_base_retrieval_chain } | tax_base_equation_chain

def get_tax_base_equation(state: AgentState) -> str:
    question = "주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요."
    tax_base_equation = tax_base_chain.invoke(question)

    print(f"tax_base_equation: {tax_base_equation}")
    return {"tax_base_equation": tax_base_equation}



# %%
# get_tax_base_equation({})

# {'tax_base_equation': '과세표준 = (Σ(주택 공시가격) - 공제금액) × 공정시장가액비율'}

# %%
tax_deduction_retrieval_chain = (
    {"context": retriver, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState) -> str:
    question = "주택에 대한 종합부동산세 계산시 공제금액을 알려주세요."
    tax_deduction = tax_deduction_retrieval_chain.invoke(question)

    print(f"tax_deduction: {tax_deduction}")
    return {"tax_deduction": tax_deduction}

# %%
# get_tax_deduction({})

# %%
from langchain_community.tools import TavilySearchResults
from datetime import date

tavliy_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ("system", "아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext: {context}"),
    ("human", "{question}")
])

tax_market_ratio_chain = tax_market_ratio_prompt | llm | StrOutputParser()

def get_tax_market_ratio(state: AgentState) :
    question = f'오늘 날짜 ({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?'
    search_tool = tavliy_search_tool.as_tool()
    context = search_tool.invoke({"query": question})
    market_ratio = tax_market_ratio_chain.invoke({"context": context, "question": question})

    print(f"market_ratio: {market_ratio}")
    return {"market_ratio": market_ratio}


# %%
# get_tax_market_ratio({})

# %%
tax_base_calculation_prompt = ChatPromptTemplate.from_messages([
    ("system", """주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 수식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}"""),
    ("human", "사용자의 주택 공시가격 정보: {question}")
])

def calculate_tax(state: AgentState) :
    tax_base_equation = state["tax_base_equation"]
    tax_deduction = state["tax_deduction"]
    market_ratio = state["market_ratio"]
    question = state["question"]

    tax_base_calculation_chain = (
        tax_base_calculation_prompt 
        | llm 
        | StrOutputParser()
    )
    tax_base = tax_base_calculation_chain.invoke({
        "tax_base_equation": tax_base_equation,
        "tax_deduction": tax_deduction,
        "market_ratio": market_ratio,
        "question": question
    })

    print(f"tax_base: {tax_base}")
    return {"tax_base": tax_base}
    

# %%
initial_state = {
    'tax_base_equation': '과세표준 = (Σ(주택 공시가격) - 공제금액) × 공정시장가액비율',
    'market_ratio': '2025년 주택 공시가격 공정시장가액비율은 1주택자의 경우 공시가격 구간에 따라 43%에서 45%까지 적용되고, 다주택자나 법인의 경우에는 60%가 적용됩니다.',
    'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 1세대 1주택자는 12억 원, 그 외의 경우는 9억 원입니다.',
    'question': '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'
}

# %%
# calculate_tax(initial_state)

# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 종합 부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요 \n\n 종합부동산세 세율: {context}"),
    ("human", """과세표준과 사용자가 소지한 주택수가 아래와 같을 때 종합부동산세를 계산해주세요
    과세표준: {tax_base}
    사용자의 주택 수: {question}""")
])

tax_rate_calculation_chain = tax_rate_calculation_prompt | llm | StrOutputParser()

def calculate_tax_rate(state: AgentState) :
    question = state["question"]
    tax_base = state["tax_base"]
    context = retriver.invoke(question)
    tax_rate = tax_rate_calculation_chain.invoke({
        "context": context,
        "question": question,
        "tax_base": tax_base
    })

    print(f"answer: {tax_rate}")
    return {"answer": tax_rate}


# %%
# calculate_tax_rate({
#     'tax_base': '주어진 정보를 바탕으로 사용자의 경우 다주택자이므로 해당 조건에 맞춰 과세표준을 계산해보겠습니다.\n\n1. 주택 공시가격의 합계:\n   - 5억 원 + 10억 원 + 20억 원 = 35억 원\n\n2. 공제금액:\n   - 다주택자의 경우 공제금액은 9억 원입니다.\n\n3. 공정시장가액비율:\n   - 다주택자의 경우 60%가 적용됩니다.\n\n과세표준 계산:\n\\( \\text{과세표준} = (\\Sigma \\text{주택 공시가격} - \\text{공제금액}) \\times \\text{공정시장가액비율} \\)\n\n\\( \\text{과세표준} = (35억 원 - 9억 원) \\times 0.60 \\)\n\n\\( \\text{과세표준} = 26억 원 \\times 0.60 \\)\n\n\\( \\text{과세표준} = 15.6억 원 \\)\n\n따라서, 과세표준은 15.6억 원입니다. 이 과세표준을 바탕으로 종합부동산세를 계산하게 됩니다. 다만, 정확한 세금 금액은 세율 및 기타 세부적인 계산식에 따라 달라질 수 있습니다.',
#     'question': question
# })

# %%
graph_builder.add_node("get_tax_base_equation", get_tax_base_equation)
graph_builder.add_node("get_tax_deduction", get_tax_deduction)
graph_builder.add_node("get_tax_market_ratio", get_tax_market_ratio)
graph_builder.add_node("calculate_tax", calculate_tax)
graph_builder.add_node("calculate_tax_rate", calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "get_tax_base_equation")
graph_builder.add_edge(START, "get_tax_deduction")
graph_builder.add_edge(START, "get_tax_market_ratio")


graph_builder.add_edge("get_tax_base_equation", "calculate_tax")
graph_builder.add_edge("get_tax_deduction", "calculate_tax")
graph_builder.add_edge("get_tax_market_ratio", "calculate_tax")

graph_builder.add_edge("calculate_tax", "calculate_tax_rate")
graph_builder.add_edge("calculate_tax_rate", END)

# %%
graph = graph_builder.compile()


