import pandas as pd
from openai import OpenAI
import time
import os
import sys
import json

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
            print(f"Warning: {f} not found.")
            
    if not dfs:
        print("Error: No CSV files successfully loaded.")
        sys.exit(1)
    
    return pd.concat(dfs, ignore_index=True)

def main():
    print("Loading essays...")
    df = load_data()
    
    if "essay_text" not in df.columns:
        print("Error: 'essay_text' column not found in merged data.")
        sys.exit(1)
        
    if "grade" not in df.columns:
        df["grade"] = None
    if "grade_reason" not in df.columns:
        df["grade_reason"] = None
        
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    # Verify LM studio is running
    print("Connecting to LM Studio server at http://localhost:1234/v1...")
    try:
        client.models.list()
    except Exception as e:
        print("Error: Could not connect to LM Studio server.")
        print("Please make sure LM Studio server is running.")
        print(f"Details: {e}")
        sys.exit(1)

    system_prompt = """You are an expert college admissions counselor who has read thousands of college application essays. Grade the following essay strictly using this rubric:

A+: Exceptional. Memorable, highly specific, emotionally resonant, unique voice, leaves a lasting impression
A: Strong. Compelling story, good structure, specific details, clear personality
B: Average. Decent writing but generic, lacks specificity or strong unique voice
C: Below average. Vague, cliché, poorly structured, forgettable

Respond ONLY with valid JSON and nothing else, no explanation outside the JSON:
{"grade": "A+", "reason": "one sentence explanation"}"""

    to_grade = df["grade"].isna() | (df["grade"] == "") | (df["grade"].astype(str).str.lower() == "unknown")
    indices_to_grade = df[to_grade].index.tolist()
    
    total = len(indices_to_grade)
    print(f"Found {total} essays to grade.")
    
    if total == 0:
        print("No essays need grading. Exiting.")
        sys.exit(0)
        
    for i, idx in enumerate(indices_to_grade):
        essay = df.at[idx, "essay_text"]
        if pd.isna(essay) or len(str(essay).strip()) == 0:
            continue
            
        print(f"Grading essay {i+1}/{total}...")
        try:
            response = client.chat.completions.create(
                model="local-model", # Any string works for LM Studio
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(essay)}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            
            # extract json if markdown blocks are returned occasionally
            if content.startswith("```json"):
                content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
            elif content.startswith("```"):
                content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                    
            content = content.strip()
            
            try:
                result = json.loads(content)
                df.at[idx, "grade"] = result.get("grade", "unknown")
                df.at[idx, "grade_reason"] = result.get("reason", "")
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse JSON response for essay {i+1}. Result was: {content}")
                
        except Exception as e:
            print(f"  Error calling LM Studio API: {e}")
            print("Please ensure your LM Studio server is running properly.")
            sys.exit(1)
            
        if (i + 1) % 10 == 0 or (i + 1) == total:
            try:
                df.to_csv("all_essays_graded.csv", index=False)
                print(f"  Saved progress to all_essays_graded.csv ({i+1}/{total})")
            except Exception as e:
                print(f"  Warning: failed to save progress: {e}")
                
        # 1 second delay between calls as requested
        time.sleep(1)

    print("Grading complete. Final dataset saved to all_essays_graded.csv")

if __name__ == "__main__":
    main()
