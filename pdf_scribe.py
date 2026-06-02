import sys
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. Check arguments
if len(sys.argv) < 2:
    print("❌ Usage: python3 pdf_scribe.py <path_to_pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]

# Check if file actually exists before crashing later
if not os.path.exists(pdf_path):
    print(f"❌ Error: File '{pdf_path}' not found.")
    sys.exit(1)

print(f"📂 Loading {pdf_path}...")

# 2. Load the PDF
try:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Combined text
    full_text = "\n\n".join([page.page_content for page in pages])
    
    # SAFETY CHECK: Crude word count estimation (1 token ~= 0.75 words)
    # If the text is massive, local LLMs might cut it off or hallucinate.
    est_tokens = len(full_text) / 4 
    print(f"✅ Loaded {len(pages)} pages (Approx. {int(est_tokens)} tokens).")

except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    sys.exit(1)

# 3. Initialize Model
# Increased temperature slightly for more creative "teaching" analogies, 
# but kept low enough for factual accuracy.
llm = ChatOllama(model="llama3.2", temperature=0.3) 

# 4. Define Prompt
# Added "Output in Markdown" instruction so it looks good in editors/Notion.
template = """
You are an expert teaching assistant for a Computer Science student.
Summarize the following content from a lecture PDF. 

Format your response in valid Markdown.

Structure the summary as follows:
## 1. Core Concepts
(Explain the main topics clearly)

## 2. Key Terminology
* **Term**: Definition
* **Term**: Definition

## 3. Exam Prep
(Provide 3 potential short-answer questions and 1 long-answer question based on this text)

---
Content to summarize:
{text}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# 5. Run and Save
print("🧠 Analyzing content (this may take a moment)...")

try:
    response = chain.invoke({"text": full_text})
    
    # Print to console
    print("\n" + "="*40)
    print(response.content)
    print("="*40)

    # NEW: Save to a file for your notes
    output_filename = f"{os.path.splitext(pdf_path)[0]}_summary.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.content)
    
    print(f"\n📝 Summary saved to: {output_filename}")

except Exception as e:
    print(f"\n❌ Error during processing: {e}")
    print("Tip: If the PDF is too large, Llama 3.2 might have run out of context memory.")
