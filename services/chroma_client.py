import chromadb


class ChromaClient:
    def __init__(self, path="./chromo_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="documents_vector_db"
        )

    def store_embeddings(
        self, uniqueId, embedding: list[list[float]], metaData: list[dict]
    ):
        print(f"Storing embedding for ID: {uniqueId}")
        print(f"Embedding: {embedding}")
        self.collection.add(ids=uniqueId, embeddings=embedding, metadatas=metaData)
        print(f"Stored embedding...")

    def retrive_data(self):
        print("Retrieving data from ChromaDB...")
        return self.collection.get()

    def query_embedding(self, embedding, top_k=5):
        return self.collection.query(query_embeddings=embedding, n_results=top_k)


# if __name__ == "__main__":
#     # Example usage
#     client = ChromaClient()
#     client.store_embeddings("example_id", [[0.1, 0.2, 0.3]], [{"meta": "data"}])
#     data = client.retrive_data()
#     print(data)
