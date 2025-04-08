from ingestion_pipeline.pdf_loader import PDFLoader
from ingestion_pipeline.text_chunker import TextChunker
from services.chroma_client import ChromaClient
from services.embedding_generator import EmbeddingGenerator


class RAGIngestionPipeline:
    def __init__(self, file_path: str):
        self.loader = PDFLoader(file_path)
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.choma_client = ChromaClient()

    def run(self):
        extracted_data = self.loader.extract_text()
        chunked_data = self.chunker.split_text(extracted_data)

        batchSize = 100
        all_ids, all_embeddings, all_metadatas = [], [], []

        for idx, data in enumerate(chunked_data):
            try:
                id = f"{data['document_name']}_{data['page']}_{idx}"
                embeddings = self.embedder.get_embeddings(data["chunk_text"])
                meta_data = {
                    "document_name": data["document_name"],
                    "page": data["page"],
                    "chunk_text": data["chunk_text"],
                }

                all_ids.append(id)
                all_embeddings.append(embeddings)
                all_metadatas.append(meta_data)

                if (idx + 1) % batchSize == 0:
                    self.choma_client.store_embeddings(
                        all_ids, all_embeddings, all_metadatas
                    )
                    all_ids, all_embeddings, all_metadatas = [], [], []
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue

        if all_ids:
            try:
                self.choma_client.store_embeddings(
                    all_ids, all_embeddings, all_metadatas
                )
            except Exception as e:
                print(f"Error storing remaining embeddings: {e}")

        print("completed.")


if __name__ == "__main__":

    print("RAG Ingestion Pipeline")
    file_path = "files/clinical_skills_guidance.pdf"  # Replace with your PDF file path
    pipeline = RAGIngestionPipeline(file_path)
    pipeline.run()
