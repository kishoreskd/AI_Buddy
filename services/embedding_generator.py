import torch
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:

    # region props
    # endregion

    # region constructor

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        hf_token=None,
        embedding_model=None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.embedding_model = embedding_model or self._initialize_embedding_model()

    # endregion

    # region public methods

    def _initialize_embedding_model(self):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
            print(f"Loaded embedding model: {self.model_name}")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error initializing embedding model: {e}")

    # helpfull for createing embeddings for a single text
    def get_embeddings(self, text):
        try:
            # print(f"Generating {text} embeddings for text...")
            embeddings = self.embedding_model.embed_query(text)
            # print(f"Generated embeddings {embeddings}  for text.")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}")

    # helpful for batch processing of texts
    def get_batch_embeddings(self, texts):
        try:
            print(f"Generating embeddings for a batch of texts...")
            embeddings = [self.embedding_model.embed_query(text) for text in texts]
            print(f"Generated embeddings for batch.")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generating batch embeddings: {e}")

    def get_embedding_from_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            print(f"Generating embeddings for documents...")
            embeddings = self.embedding_model.embed_documents(texts)
            print(f"Generated embeddings for documents.")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generating document embeddings: {e}")

    # endregion
