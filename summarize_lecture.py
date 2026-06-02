from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize the model
llm = ChatOllama(model="llama3.2", temperature=0.3)

# 2. Define the Prompt Template
# This tells the model strictly how to behave
template = """
You are an expert teaching assistant. 
Your task is to summarize the following lecture transcript into concise study notes.

Format the output as:
1. **Key Concepts**: Bullet points of main ideas.
2. **Important Definitions**: Terms and their meanings.
3. **Summary**: A 2-sentence wrap-up.

Lecture Transcript:
{transcript}
"""

prompt = ChatPromptTemplate.from_template(template)

# 3. Create the Chain (Prompt -> LLM)
chain = prompt | llm

# 4. Sample Transcript (You can replace this text later)
sample_transcript = """
Okay everyone, welcome back. Today we are talking about 'Deadlocks' in Operating Systems. 
Basically, a deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another resource acquired by some other process. 
Think of it like four cars at a 4-way stop, and everyone decides to wait for the other person to go. No one moves.
There are four necessary conditions for a deadlock to happen: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait. 
If we can break just one of these conditions, we can prevent the deadlock entirely.
"""

print("Generating lecture notes...")

# 5. Run the chain
response = chain.invoke({"transcript": sample_transcript})

print("\n" + "="*40)
print(response.content)
print("="*40)
