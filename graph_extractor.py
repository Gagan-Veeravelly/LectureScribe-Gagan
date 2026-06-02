import os
import json
from openai import OpenAI

# --- OLLAMA SETUP ---
# Pointing to your local Ollama server
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)

# ✅ Updated to match your installed model
MODEL_NAME = "llama3.2" 

SYSTEM_PROMPT = """
You are a strict Data Extraction Engine. 
You are NOT a chatbot. Do not apologize. Do not explain.
Your ONLY output is valid JSON.

Task: Extract a Knowledge Graph from the provided text.
Format:
{
  "triples": [
    {"head": "Entity1", "relation": "relationship", "tail": "Entity2"},
    {"head": "Entity2", "relation": "relationship", "tail": "Entity3"}
  ]
}

Rules:
1. "head" and "tail" must be short concepts (e.g., "Deadlock", "Mutex").
2. "relation" must be a verb (e.g., "causes", "requires").
3. Extract at least 5-10 triples.
"""

def extract_triples(text_chunk):
    print(f"🦙 Asking Ollama ({MODEL_NAME}) to extract triples...")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"TEXT TO ANALYZE:\n{text_chunk}"}
            ],
            temperature=0, 
            response_format={"type": "json_object"} 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return "{}"

# --- Test Data (From your OS Lecture PDF) ---
test_text = """
Deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another resource acquired by some other process.
Resource Allocation Graphs (RAG) and Wait-For Graphs relate to Deadlocks.
Dekker's Algorithm is a solution to the Critical Section Problem (CSP).
The three requirements for solving CSP are Mutual Exclusion, Progress, and Bounded Waiting.
Peterson's Algorithm is another solution to CSP.
"""

if __name__ == "__main__":
    # 1. Extract
    json_response = extract_triples(test_text)
    
    # 2. Parse & Validate
    try:
        # Clean potential markdown formatting
        clean_json = json_response.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0]

        data = json.loads(clean_json)
        
        # Handle variations in JSON structure
        triples = data.get("triples", [])
        if not triples and isinstance(data, list):
            triples = data 
            
        # 3. Save
        if triples:
            with open("graph_data.json", "w") as f:
                json.dump(triples, f, indent=4)
            print(f"✅ Success! Extracted {len(triples)} triples.")
            print("Check graph_data.json to see the results.")
        else:
            print("⚠️ Parsed JSON but found no triples. Output was:", data)

    except json.JSONDecodeError as e:
        print("❌ JSON Error. The model might have outputted chat text.")
        print("Raw Output:", json_response)
