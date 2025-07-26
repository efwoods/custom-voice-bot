import json
from tqdm import tqdm
from pathlib import Path
import argparse

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vectorstore(input_file: str, persist_dir: str):
    texts = []
    metadatas = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading mom messages"):
            o = json.loads(line)
            texts.append(o["text"])
            metadatas.append({k: v for k, v in o.items() if k != "text"})

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_texts(
        texts=texts,
        embedding=embed_model,
        metadatas=metadatas,
        persist_directory=persist_dir,
    )
    db.persist()
    print(f"Vectorstore persisted to {persist_dir} with {len(texts)} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="data/processed/mom_clean.jsonl"
    )
    parser.add_argument("--persist_dir", type=str, default="data/processed/chroma_mom")
    args = parser.parse_args()
    build_vectorstore(args.input_file, args.persist_dir)
