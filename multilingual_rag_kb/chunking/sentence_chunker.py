def sentence_chunk(sentences, chunk_size=5):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks
