"""
build_humanizer_index.py
────────────────────────
Creates the `human_patterns` ChromaDB collection for Agent 5 (The Humanizer).

Unlike the main `essays` collection (which stores whole essays), this collection
stores individual *sentences* extracted from high-quality essays. This enables
sentence-level similarity search so the Humanizer can ground its rewrites in
real human sentence DNA.

Data sources:
  • openessays_essays.csv       — all essays (accepted to top schools)
  • essaysthatworked_essays.csv — only A / A+ graded essays
  • EssayFroum-Dataset.csv      — all essays (curated forum dataset)

Usage:
    python build_humanizer_index.py
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
import re

# Force UTF-8 for print() on Windows (avoids cp1252 UnicodeEncodeError for emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Sentence splitting ─────────────────────────────────────────────────────
# Simple regex-based splitter that avoids nltk dependency.
# Splits on . ! ? followed by whitespace + uppercase letter (new sentence signal).

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []

    # Split on sentence-ending punctuation followed by space + uppercase
    # This handles most essay text reliably.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text)

    sentences = []
    for part in parts:
        s = part.strip()
        # Keep only sentences with at least 40 characters (filters out fragments)
        if len(s) >= 40:
            sentences.append(s)

    return sentences


def load_and_prepare_data() -> pd.DataFrame:
    """Load essay data from all CSV sources and return a unified DataFrame."""
    records = []

    # ── 1. openessays_essays.csv (all are accepted essays → include all) ────
    if os.path.exists("openessays_essays.csv"):
        try:
            df = pd.read_csv("openessays_essays.csv")
            for _, row in df.iterrows():
                text = str(row.get("essay_text", "")).strip()
                if len(text) >= 200:
                    records.append({
                        "text": text,
                        "source": "openessays",
                        "school": str(row.get("school", "unknown")).strip() or "unknown",
                        "grade": "accepted"
                    })
            print(f"  ✅ openessays_essays.csv: {sum(1 for r in records if r['source']=='openessays')} essays loaded")
        except Exception as e:
            print(f"  ⚠️  Error reading openessays_essays.csv: {e}")
    else:
        print("  ⚠️  openessays_essays.csv not found — skipping")

    # ── 2. essaysthatworked_essays.csv (only A and A+ grades) ───────────────
    if os.path.exists("essaysthatworked_essays.csv"):
        try:
            df = pd.read_csv("essaysthatworked_essays.csv")
            for _, row in df.iterrows():
                grade = str(row.get("grade", "")).strip().upper()
                if grade in ("A", "A+"):
                    text = str(row.get("essay_text", "")).strip()
                    if len(text) >= 200:
                        records.append({
                            "text": text,
                            "source": "essaysthatworked",
                            "school": str(row.get("school", "unknown")).strip() or "unknown",
                            "grade": grade
                        })
            print(f"  ✅ essaysthatworked_essays.csv: {sum(1 for r in records if r['source']=='essaysthatworked')} A/A+ essays loaded")
        except Exception as e:
            print(f"  ⚠️  Error reading essaysthatworked_essays.csv: {e}")
    else:
        print("  ⚠️  essaysthatworked_essays.csv not found — skipping")

    # ── 3. EssayFroum-Dataset.csv (forum-curated → include all) ─────────────
    if os.path.exists("EssayFroum-Dataset.csv"):
        try:
            df = pd.read_csv("EssayFroum-Dataset.csv")
            # This CSV uses "Cleaned Essay" or "Correct Grammar" columns
            text_col = "Cleaned Essay" if "Cleaned Essay" in df.columns else "Correct Grammar"
            for _, row in df.iterrows():
                text = str(row.get(text_col, "")).strip()
                if len(text) >= 200:
                    records.append({
                        "text": text,
                        "source": "essayforum",
                        "school": "forum",
                        "grade": "curated"
                    })
            print(f"  ✅ EssayFroum-Dataset.csv: {sum(1 for r in records if r['source']=='essayforum')} essays loaded")
        except Exception as e:
            print(f"  ⚠️  Error reading EssayFroum-Dataset.csv: {e}")
    else:
        print("  ⚠️  EssayFroum-Dataset.csv not found — skipping")

    if not records:
        print("\n❌ No essays loaded. Ensure at least one CSV file is in the current directory.")
        sys.exit(1)

    print(f"\n📊 Total source essays: {len(records)}")
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  Agent 5 Humanizer — Building sentence-level index")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n📂 Loading CSV data sources...")
    df = load_and_prepare_data()

    # ── Split into sentences ─────────────────────────────────────────────────
    print("\n✂️  Splitting essays into sentences...")
    sentence_records = []
    for idx, row in df.iterrows():
        sentences = split_sentences(row["text"])
        for sent_idx, sentence in enumerate(sentences):
            sentence_records.append({
                "sentence": sentence,
                "source": row["source"],
                "school": row["school"],
                "grade": row["grade"],
                "essay_index": int(idx),
                "sentence_index": sent_idx
            })

    print(f"   Total sentences extracted: {len(sentence_records)}")
    if not sentence_records:
        print("❌ No sentences extracted. Check your CSV data.")
        sys.exit(1)

    # ── Load embedding model ─────────────────────────────────────────────────
    print("\n🧠 Loading embedding model 'all-MiniLM-L6-v2'...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # ── Initialize ChromaDB ──────────────────────────────────────────────────
    print("\n💾 Initializing ChromaDB collection 'human_patterns'...")
    try:
        client = chromadb.PersistentClient(path="essay_db")
        # Delete existing collection to rebuild cleanly
        try:
            client.delete_collection("human_patterns")
            print("   (Deleted existing 'human_patterns' collection)")
        except Exception:
            pass
        collection = client.create_collection(name="human_patterns")
    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        sys.exit(1)

    # ── Embed and index in batches ───────────────────────────────────────────
    print(f"\n🔄 Embedding and indexing {len(sentence_records)} sentences...")
    texts = [r["sentence"] for r in sentence_records]
    batch_size = 256

    total_indexed = 0
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_ids = [f"sent_{j}" for j in range(i, end_idx)]
        batch_metadatas = [
            {
                "source": sentence_records[j]["source"],
                "school": sentence_records[j]["school"],
                "grade": sentence_records[j]["grade"],
                "essay_index": sentence_records[j]["essay_index"],
                "sentence_index": sentence_records[j]["sentence_index"]
            }
            for j in range(i, end_idx)
        ]

        try:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            total_indexed += len(batch_texts)
            print(f"   Indexed {total_indexed}/{len(texts)} sentences")
        except Exception as e:
            print(f"❌ Error indexing batch {i}-{end_idx}: {e}")
            sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Humanizer index built successfully!")
    print("=" * 60)
    print(f"  Collection: human_patterns")
    print(f"  Total sentences indexed: {total_indexed}")
    print(f"  Sources:")
    for src in df["source"].unique():
        count = sum(1 for r in sentence_records if r["source"] == src)
        print(f"    • {src}: {count} sentences")
    print(f"\n  ChromaDB path: ./essay_db")
    print(f"  Ready for Agent 5 (The Humanizer) in app.py\n")


if __name__ == "__main__":
    main()
