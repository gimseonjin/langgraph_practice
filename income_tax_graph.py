# %%
# vector store 생성
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection",
    embedding_function=embeddings
)

retriver = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
# llm 설정
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# %%
# StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    answer: str
    context: List[Document]

# %%
# graph builder
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

# %%
# retriver node
def retrive(state: AgentState) -> AgentState:
    question = state["question"]
    docs = retriver.invoke(question)
    return {"context": docs}

# %%
from langchain_classic import hub

generate_prompt = hub.pull("rlm/rag-prompt")

def generate(state: AgentState) -> AgentState:
    content = state["context"]
    question = state["question"]
    rag_chain = generate_prompt | llm
    answer = rag_chain.invoke({"context": content, "question": question})

    print(f"answer: {answer.content}")
    return {"answer": answer.content}

# %%
# chekc docs node
from langchain_classic import hub
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def relevance_doc(state: AgentState) -> Literal['relevant', 'irrelevant']:
    question = state["question"]
    context = state["context"]
    doc_relevance_chain = doc_relevance_prompt | llm
    relevance = doc_relevance_chain.invoke({"question": question, "documents": context})

    print(f"context: {context}")
    print(f"relevance: {relevance}")

    if relevance['Score'] == 1:
        return 'relevant'
    
    return 'irrelevant'


# %%
# rewrite node
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ["사람과 관련된 표현 -> 거주자", "세금과 관련된 표현 -> 누진세 포함"]

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
사전: {dictionary}
질문: {{question}}
""")

def rewrite(state: AgentState):
    question = state["question"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten_question = rewrite_chain.invoke({"question": question})

    print(f"rewritten_question: {rewritten_question}")
    return {"question": rewritten_question}

# %%
# hallucination node
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template(f"""
You are a theache tasked with evaluation whether a studen's answer is based on facts or not,
Chech whether ther student's anser is hallucianted or not
given context, which are excerpts from income tax law, and a student's answer,
If the student's answer is based on facts, respond with "not hlluciated"
If the student's answer is not based on facts, respond with "hallucinated"
학생의 답변: {{student_answer}}
documents: {{documents}}
""")

hallucination_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def chekc_hallucination(state: AgentState) -> Literal['not hallucinated', 'hallucinated']:
    answer = state["answer"]
    context = state["context"]
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({"student_answer": answer, "documents": context})

    print(f"context: {context}")
    print(f"response: {response}")
    
    return response

    


# %%
from langchain_classic import hub
helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState):
    question = state["question"]
    answer = state["answer"]
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({"question": question, "student_answer": answer})

    print(f"helpfulness: {response}")

    if response['Score'] == 1:
        return 'helpful'
    
    return 'not helpful'
    
    
def check_helpfulness(state: AgentState) -> Literal['helpful', 'not helpful']:
    return state

# %%
graph_builder.add_node("retrive", retrive)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("helpfulness", check_helpfulness)


# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "retrive")
graph_builder.add_conditional_edges(
    "retrive",
    relevance_doc,
    {
        "relevant": "generate",
        "irrelevant": END
    }
)

graph_builder.add_conditional_edges(
    "generate",
    chekc_hallucination,
    {
        "not hallucinated": "helpfulness",
        "hallucinated": "generate"
    }
)

graph_builder.add_conditional_edges(
    "helpfulness",
    check_helpfulness_grader,
    {
        "helpful": END,
        "not helpful": 'rewrite'
    }
)

graph_builder.add_edge("rewrite", "retrive")


# %%
graph = graph_builder.compile()