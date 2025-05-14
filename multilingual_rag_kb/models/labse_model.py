from sentence_transformers import SentenceTransformer

class LaBSEEmbedder:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    