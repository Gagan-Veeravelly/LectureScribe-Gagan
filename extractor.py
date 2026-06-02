import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Schema Definition ---
class KnowledgeTriple(BaseModel):
    subject: str = Field(description="The entity that is the subject")
    predicate: str = Field(description="The relationship between subject and object")
    object: str = Field(description="The entity that is the object")

class KnowledgeGraph(BaseModel):
    triples: List[KnowledgeTriple] = Field(description="List of extracted relationships")

# --- Prompt ---
system_prompt = """
You are an expert Data Scientist. 
Extract a strict Knowledge Graph from the text.
1. Identify technical entities.
2. Identify relationships.
3. Output STRICT JSON matching the schema.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{text}"),
])

structured_llm = llm.with_structured_output(KnowledgeGraph)
extraction_chain = prompt | structured_llm

def extract_graph_from_text(text_chunk):
    print("🧠 Processing text with LLM...")
    try:
        return extraction_chain.invoke({"text": text_chunk})
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

if __name__ == "__main__":
    sample_text = "Neural Networks are a subset of Machine Learning."
    result = extract_graph_from_text(sample_text)

    if result:
        print("\n--- Extracted Knowledge Graph ---")
        for item in result.triples:
            print(f"({item.subject}) --[{item.predicate}]--> ({item.object})")
