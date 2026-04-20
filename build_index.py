import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys

def load_data():
    csv_files = ["openessays_essays.csv", "collegebase_essays.csv", "essaysthatworked_essays.csv"]
    dfs = []
    for f in csv_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        else:
            print(f"Warning: {f} not found in current directory.")
            
    if not dfs:
        print("Error: No CSV files successfully loaded. Please ensure in current directory.")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    return df

def main():
    print("Loading valid csv files...")
    df = load_data()
    
    if "essay_text" not in df.columns:
        print("Error: 'essay_text' column not found in data.")
        sys.exit(1)
        
    df["essay_text"] = df["essay_text"].fillna("").astype(str)
    
    # Drop rows where empty or under 100 characters
    initial_len = len(df)
    df = df[df["essay_text"].str.strip().str.len() >= 100].copy()
    final_len = len(df)
    print(f"Dropped {initial_len - final_len} essays that are empty or under 100 characters.")
    
    if len(df) == 0:
        print("Error: No valid essays left to index. Exiting.")
        sys.exit(1)
        
    print("Loading embedding model 'all-MiniLM-L6-v2' (this may download weights on first run)...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading sentence-transformers model: {e}")
        sys.exit(1)
        
    print("Initializing ChromaDB at './essay_db'...")
    try:
        client = chromadb.PersistentClient(path="essay_db")
        # To avoid adding duplicate embeddings if run multiple times, 
        # we can delete the existing collection or use get_or_create_collection
        try:
            client.delete_collection("essays")
        except:
            pass
            
        collection = client.create_collection(name="essays")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)
        
    # Prepare data for Chroma
    texts = df["essay_text"].tolist()
    
    def safe_str(val):
        if pd.isna(val):
            return "unknown"
        s = str(val).strip()
        return s if s else "unknown"
        
    schools = df.get("school", pd.Series(["unknown"] * len(df))).apply(safe_str).tolist()
    grades = df.get("grade", pd.Series(["unknown"] * len(df))).apply(safe_str).tolist()
    types = df.get("essay_type", pd.Series(["unknown"] * len(df))).apply(safe_str).tolist()
    
    print(f"Indexing {len(texts)} essays...")
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_ids = [f"essay_{j}" for j in range(i, end_idx)]
        batch_metadatas = [
            {
                "school": schools[j],
                "grade": grades[j],
                "essay_type": types[j]
            }
            for j in range(i, end_idx)
        ]
        
        try:
            # We must convert embeddings to lists for ChromaDB
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"Error indexing batch {i} to {end_idx}: {e}")
            sys.exit(1)
            
        print(f"Indexed {end_idx}/{len(texts)} essays")
        
    print("\n--- Indexing Complete ---")
    print(f"Total essays indexed: {len(df)}")
    
    print("\nBreakdown by essay type:")
    for essay_type, count in df.get("essay_type", pd.Series()).fillna("unknown").value_counts().items():
        print(f"  {essay_type}: {count}")
        
    print("\nBreakdown by grade:")
    for grade, count in df.get("grade", pd.Series()).fillna("unknown").value_counts().items():
        print(f"  {grade}: {count}")

if __name__ == "__main__":
    main()
