import sys
import traceback
import os 
from multilingual_rag_kb.chunking.fixed_overlap_chunker import fixed_length_chunk
from multilingual_rag_kb.chunking.sentence_chunker import sentence_chunk

RAW_DIR = "multilingual_rag_kb/data/raw"
PROCESSED_DIR = "multilingual_rag_kb/data/processed"


def chunk_demo():
    for file_name in os.listdir(PROCESSED_DIR):
        if not file_name.endswith("_clean.txt"):
            continue

        print(f"\n Processing: {file_name}")
        with open(os.path.join(PROCESSED_DIR, file_name), 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines() if line.strip()]

        fixed_chunks = fixed_length_chunk(sentences, chunk_size=5, overlap=2)
        sent_chunks = sentence_chunk(sentences, chunk_size=5)

        print(f" Fixed Chunks (5, overlap 2): {len(fixed_chunks)}")
        print(f" Sentence Chunks (5 each): {len(sent_chunks)}")
        print("Sample Fixed Chunk:\n", fixed_chunks[0][:300])
        print("Sample Sentence Chunk:\n", sent_chunks[0][:300])

if __name__ == "__main__":
    # preprocess_all_files()
    chunk_demo()