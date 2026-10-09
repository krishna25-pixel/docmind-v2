"""
llm.py — LLM wrapper (Groq). Swappable by changing LLM_MODEL in config.py.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, LLM_TEMPERATURE
from src.generation.prompts import RAG_SYSTEM_PROMPT


def build_chain(groq_api_key=None, model: str = LLM_MODEL):
    """Return a runnable: dict(context, history, question) -> str."""
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing. Set in .env or pass explicitly.")

    llm = ChatGroq(api_key=api_key, model=model, temperature=LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
