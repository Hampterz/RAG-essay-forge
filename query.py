import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sys

def main():
    print("Initializing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path="essay_db")
        collection = client.get_collection(name="essays")
    except Exception as e:
        print(f"Error loading ChromaDB from 'essay_db': {e}")
        print("Make sure you have run build_index.py first so the database is setup.")
        sys.exit(1)
        
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("Please paste your essay below.")
    print("Type 'END' on a new line and press Enter when finished:")
    print("="*50)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
            
    user_essay = "\n".join(lines).strip()
    if not user_essay:
        print("Error: Empty essay provided. Exiting.")
        sys.exit(1)
        
    print("\nEncoding your essay and searching for similar accepted essays...")
    try:
        user_embedding = model.encode(user_essay, show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=[user_embedding],
            n_results=5
        )
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        sys.exit(1)
        
    if not results['documents'] or len(results['documents'][0]) == 0:
        print("Error: No similar essays were found in the database. Ensure build_index.py populated data.")
        sys.exit(1)
        
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    
    formatted_references = ""
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
        school = meta.get("school", "unknown")
        grade = meta.get("grade", "unknown")
        essay_type = meta.get("essay_type", "unknown")
        formatted_references += f"--- Reference Essay {i+1} ---\n"
        formatted_references += f"School: {school} | Grade: {grade} | Type: {essay_type}\n"
        formatted_references += f"{doc}\n\n"
        
    system_prompt_template = """You are an expert college admissions counselor. A student wants honest, detailed feedback on their college application essay.

Here are 5 real essays from accepted students for reference:

{retrieved_essays}

Now give detailed feedback on the student's essay below. Be specific, honest, and constructive.

Student essay:
{user_essay}

Your feedback should cover:
1. Overall grade (A+/A/B/C) and a clear reason why
2. What is working well — be specific, quote parts of their essay
3. What needs improvement — be honest
4. 3 to 5 concrete actionable suggestions
5. How it compares to the reference essays"""

    prompt = system_prompt_template.format(
        retrieved_essays=formatted_references,
        user_essay=user_essay
    )
    
    print("Connecting to LM Studio to generate feedback...")
    openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    try:
        # We start to stream immediately
        response = openai_client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.7,
            stream=True
        )
        
        print("\n" + "*"*60)
        print("ADMISSIONS COUNSELOR FEEDBACK")
        print("*"*60 + "\n")
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError communicating with LM Studio: {e}")
        print("Please ensure your LM Studio server is actually running at http://localhost:1234/v1")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
