from operator import itemgetter
from typing import Dict, List, Optional, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Polars.

Generate a comprehensive and informative answer of 800 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

VECTORDB_PATH = "/chroma/trinosql"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def get_retriever() -> BaseRetriever:
    return Chroma(persist_directory=VECTORDB_PATH, embedding_function=get_embeddings_model()).as_retriever(search_type="mmr")


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))),
            conversation_chain,
        ),
        (
            RunnableLambda(itemgetter("question"))
            | retriever
        )
    )

def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    )
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = (prompt | llm | StrOutputParser())

    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
retriever = get_retriever()
answer_chain = create_chain(llm, retriever)
