import sys
import traceback
import os 
from multilingual_rag_kb.chunking.fixed_overlap_chunker import fixed_length_chunk
from multilingual_rag_kb.chunking.sentence_chunker import sentence_chunk
from multilingual_rag_kb.utils.text_cleaner import preprocess_file
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

RAW_DIR = "multilingual_rag_kb/data/raw"
PROCESSED_DIR = "multilingual_rag_kb/data/processed"

# def preprocess_all_files():
#     for file_name in os.listdir(RAW_DIR):
#         if not file_name.endswith(".txt"):
#             continue

#         input_path = os.path.join(RAW_DIR, file_name)
#         output_path = os.path.join(PROCESSED_DIR, file_name.replace(".txt", "_clean.txt"))

#         print(f"Preprocessing: {file_name}")
#         preprocess_file(input_path, output_path)

# import re
# import nltk
# from nltk import word_tokenize, pos_tag, ne_chunk
# from nltk.tree import Tree
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')



# # --- Date Regex ---
# date_pattern = re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b")

# def contains_date(text):
#     return bool(date_pattern.search(text))

# # # --- Entity Extractor ---
# def extract_named_entities(text):
#     try:
#         entities = []
#         for sentence in nltk.sent_tokenize(text):
#             tree = ne_chunk(pos_tag(word_tokenize(sentence)))
#             sentence_entities = [" ".join(c[0] for c in subtree) for subtree in tree if isinstance(subtree, Tree)]
#             entities.extend(sentence_entities)
#         return list(set(entities))
#     except:
#         return []





# def chunk_demo():
#     for file_name in os.listdir(PROCESSED_DIR):
#         if not file_name.endswith("_clean.txt"):
#             continue

#         print(f"\n Processing: {file_name}")
#         with open(os.path.join(PROCESSED_DIR, file_name), 'r', encoding='utf-8') as f:
#             sentences = [line.strip() for line in f.readlines() if line.strip()]

#         fixed_chunks = fixed_length_chunk(sentences, chunk_size=5, overlap=2)
#         sent_chunks = sentence_chunk(sentences, chunk_size=5)

#         print(f" Fixed Chunks (5, overlap 2): {len(fixed_chunks)}")
#         print(f" Sentence Chunks (5 each): {len(sent_chunks)}")
#         print("Sample Fixed Chunk:\n", fixed_chunks[0][:300])
#         print("Sample Sentence Chunk:\n", sent_chunks[0][:300])

# if __name__ == "__main__":
#     # preprocess_all_files()
#     chunk_demo()
from multilingual_rag_kb.models.e5_model import E5Embedder
from multilingual_rag_kb.models.sbert_model import SBERTEmbedder
from multilingual_rag_kb.models.labse_model import LaBSEEmbedder
from multilingual_rag_kb.vector_store.pinecone_store import PineconeStore
from multilingual_rag_kb.config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
from multilingual_rag_kb.models.e5_model import E5Embedder
# def embed_chunks_example():
#     sbert = SBERTEmbedder()
#     labse = LaBSEEmbedder()
#     e5 = E5Embedder()

#     for file_name in os.listdir(PROCESSED_DIR):
#         if not file_name.endswith("_clean.txt"):
#             continue

#         with open(os.path.join(PROCESSED_DIR, file_name), 'r', encoding='utf-8') as f:
#             sentences = [line.strip() for line in f.readlines() if line.strip()]

#         chunks = sentence_chunk(sentences, chunk_size=5)

#         print(f"\n Embedding {len(chunks)} chunks from {file_name}")
#         print("SBERT (first vector):", sbert.embed([chunks[0]])[0][:5])
#         print("LaBSE (first vector):", labse.embed([chunks[0]])[0][:5])
#         print("E5 (first vector):", e5.embed([chunks[0]])[0][:5])

# def push_to_pinecone():
#     embedder = E5Embedder()
#     store = PineconeStore(PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV, namespace="finance_rag_v2")


#     for file_name in os.listdir(PROCESSED_DIR):
#         if not file_name.endswith("_clean.txt"):
#             continue

#         with open(os.path.join(PROCESSED_DIR, file_name), 'r', encoding='utf-8') as f:
#             sentences = [line.strip() for line in f.readlines() if line.strip()]

#         chunks = sentence_chunk(sentences, chunk_size=5)
#         embeddings = embedder.embed(chunks)

#         metadata = [
#             {
#                 "source": file_name.replace("_clean.txt", ""),
#                 "chunk_id": i,
#                 "text": chunks[i],
#                 "entities": extract_named_entities(chunks[i]),
#                 "has_date": contains_date(chunks[i])
#     }
#             for i in range(len(chunks))
#         ]

        
#         store.upsert_chunks(chunks, embeddings, metadata)
#         print(f"Uploaded {len(chunks)} chunks from {file_name} to Pinecone.")

# def run_similarity_search():
#     query = input("\n Enter your finance-related question:\n> ")

#     embedder = E5Embedder()
#     store = PineconeStore(
#         PINECONE_INDEX_NAME,
#         PINECONE_API_KEY,
#         PINECONE_ENV,
#         namespace="finance_rag_v2"  # your latest namespace
#     )

#     query_embedding = embedder.embed([f"query: {query}"])[0]

#     # filter_metadata = {
#     #     "entities": {"$in": ["Sheila A. Stamps"]},
#     #     "has_date": True
#     # }

#     # For debugging: remove filter, get all top_k results
#     results = store.search(query_embedding, top_k=5)

#     print("\n Top Matching Chunks:")
#     if not results.get('matches'):
#         print("No matches found.")
#     else:
#         for match in results['matches']:
#             metadata = match['metadata']
#             print(f"\n[Score: {match['score']:.4f}] From: {metadata['source']}, Chunk ID: {metadata['chunk_id']}")
#             print(f"→ Content Preview: {metadata.get('text', 'N/A')[:200]}")
#             print(f"→ Entities: {metadata.get('entities', [])}")
#             print(f"→ Has Date: {metadata.get('has_date', False)}")




from multilingual_rag_kb.llm.prompt_templates import build_rag_prompt
from multilingual_rag_kb.llm.ollama_engine import get_llm_response
from multilingual_rag_kb.llm.gemini_engine import get_gemini_response

def run_rag_chat():
    query = input("\n Ask a question (multilingual supported):\n> ")
    embedder = E5Embedder()
    store = PineconeStore(PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV,namespace='finance_rag_v2')

    query_embedding = embedder.embed([f"query: {query}"])[0]
    
#     filter_metadata = {
#     "entities": {"$in": ["R. Brad Oates"]},
#     "has_date": True
# }

    result = store.search(query_embedding, top_k=5)

    

    top_chunks = [match["metadata"]["text"] for match in result["matches"]]
    print("\n Retrieved Chunks:")
    for i, c in enumerate(top_chunks):
        print(f"[{i+1}] {c[:200]}")

    prompt = build_rag_prompt(query, top_chunks)
    # print("\nPrompt Sent to LLM:\n")
    # print(prompt)
    
    llm_choice = input("\nChoose LLM engine ([O]llama/GenAI/[g]Gemini, default=Ollama): ").strip().lower()
    if llm_choice == 'g':
        answer =  get_gemini_response(prompt)
    
    else:
        answer = get_llm_response(prompt)

    print("\n LLM Answer:")
    print(answer)
    
def evaluate_top_k_accuracy(test_queries, ground_truths, k=5):
    """
    test_queries: list of user queries
    ground_truths: list of expected relevant chunk texts (same order as test_queries)
    k: number of top chunks to consider
    """
    correct = 0
    for query, truth in zip(test_queries, ground_truths):
        embedder = E5Embedder()
        store = PineconeStore(PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV, namespace='finance_rag_v2')
        query_embedding = embedder.embed([f"query: {query}"])[0]
        result = store.search(query_embedding, top_k=k)
        top_chunks = [match["metadata"]["text"] for match in result["matches"]]
        if any(truth in chunk for chunk in top_chunks):
            correct += 1
    accuracy = correct / len(test_queries)
    print(f"Top-{k} Retrieval Accuracy: {accuracy:.2%}")
    return accuracy

def evaluate_relevance(test_queries, reference_answers, llm_func, k=5):
    """
    llm_func: function to get LLM answer (e.g., get_llm_response or get_gemini_response)
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scores = []
    for query, ref in zip(test_queries, reference_answers):
        embedder = E5Embedder()
        store = PineconeStore(PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV, namespace='finance_rag_v2')
        query_embedding = embedder.embed([f"query: {query}"])[0]
        result = store.search(query_embedding, top_k=k)
        top_chunks = [match["metadata"]["text"] for match in result["matches"]]
        prompt = build_rag_prompt(query, top_chunks)
        answer = llm_func(prompt)
        answer_emb = model.encode([answer])[0]
        ref_emb = model.encode([ref])[0]
        sim = cosine_similarity([answer_emb], [ref_emb])[0][0]
        scores.append(sim)
        print(f"Query: {query}\nLLM Answer: {answer}\nReference: {ref}\nSimilarity: {sim:.2f}\n")
    avg_score = sum(scores) / len(scores)
    print(f"Average Relevance (cosine similarity): {avg_score:.2f}")
    return avg_score

def measure_response_latency(query, llm_func, k=5):
    embedder = E5Embedder()
    store = PineconeStore(PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV, namespace='finance_rag_v2')
    query_embedding = embedder.embed([f"query: {query}"])[0]
    result = store.search(query_embedding, top_k=k)
    top_chunks = [match["metadata"]["text"] for match in result["matches"]]
    prompt = build_rag_prompt(query, top_chunks)
    start = time.time()
    answer = llm_func(prompt)
    end = time.time()
    latency = end - start
    print(f"Response latency: {latency:.2f} seconds")
    return latency

# Example test data (replace with your real data)
test_queries = [
    " tell me about Apple Seven Advisors",
    "What is they raised money?"
]
ground_truths = [
    """pple REIT Seven, Inc. (“AR7”), is a non-traded REIT organized under the laws of Virginia with a class of securities registered under Section 12(g) of the Exchange Act and is subject to the reporting requirements of Section 13(a) of the Exchange Act.
AR7 owned 51 hotels operating in 18 states as of December 31, 2012"""
]
reference_answers = [
    """AR7 raised approximately $1 billion between May 2005 and July 2007, and had approximately 19,800 beneficial unitholders as of February 28, 2013.
AR7 instituted a DRIP by filing a Form S-3 registration statement on July 17, 2007, which it amended and restated on January 13, 2012.
As of December 31, 2012, AR7 had sold approximately 11.3 million units representing approximately $124.5 million in proceeds pursuant to these DRIP offerings."""
]

def main_menu():
    print("\nSelect an option:")
    print("1. Run RAG Chat")
    print("2. Evaluate Top-k Retrieval Accuracy")
    print("3. Evaluate Relevance of Generated Answers")
    print("4. Measure Response Latency")
    print("5. Exit")
    choice = input("> ").strip()
    if choice == '1':
        run_rag_chat()
    elif choice == '2':
        evaluate_top_k_accuracy(test_queries, ground_truths, k=5)
    elif choice == '3':
        # Choose LLM engine for evaluation
        llm_choice = input("Use [O]llama or [G]emini for LLM? (default=Ollama): ").strip().lower()
        if llm_choice == 'g':
            llm_func = get_gemini_response
        else:
            llm_func = get_llm_response
        evaluate_relevance(test_queries, reference_answers, llm_func, k=5)
    elif choice == '4':
        query = input("Enter a query to measure latency: ")
        llm_choice = input("Use [O]llama or [G]emini for LLM? (default=Ollama): ").strip().lower()
        if llm_choice == 'g':
            llm_func = get_gemini_response
        else:
            llm_func = get_llm_response
        measure_response_latency(query, llm_func, k=5)
    elif choice == '5':
        print("Exiting.")
        exit()
    else:
        print("Invalid choice.")
        main_menu()

if __name__ == "__main__":
    # preprocess_all_files()
    # chunk_demo()
    # embed_chunks_example()
    # push_to_pinecone()
    # run_similarity_search()
    run_rag_chat()
    main_menu()