from gettext import find
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Optional
import string

# from pdf_loader import PDFLoader

# from pdf_loader import PDFLoader
# import json

try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class TextChunker:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):

        # self.download_datasets()
        self.stopwords = set(stopwords.words("english"))

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, extracted_data: list[dict]) -> list[dict]:
        chunked_data = []
        for data in extracted_data:
            chunks = self.splitter.split_text(data["text"])
            for chunk in chunks:
                cleaned_chunk = self.clean_text(chunk)
                if cleaned_chunk:
                    chunked_data.append(
                        {
                            "chunk_text": cleaned_chunk,
                            "page": data["page"],
                            "document_name": data["document_name"],
                        }
                    )
        return chunked_data

    def clean_text(
        self, text: str, remove_stopwords: bool = False, min_char_length: int = 50
    ):

        # Remove headers/footers or page references
        text = re.sub(r"Page\s+\d+|Confidential|Copyright.*|\n{2,}", " ", text)

        # Remove special characters except periods and common punctuation
        text = re.sub(r"[^\w\s.,;:!?()-]", "", text)

        # Optionally remove stopwords
        if remove_stopwords:
            words = word_tokenize(text)
            words = [word for word in words if word.lower() not in self.stopwords]
            text = " ".join(words)

        # Final length check
        if len(text) < min_char_length:
            return None

        return text.strip()


# if __name__ == "__main__":
#     pdf_loader = PDFLoader("files/clinical_skills_guidance.pdf")
#     chunker = TextChunker()
#     chunked_datas = chunker.split_text(pdf_loader.extract_text())

#     with open("chunked_data.json", "w", encoding="utf-8") as json_file:
#         json.dump(chunked_datas, json_file, ensure_ascii=False, indent=4)

#     print("Chunked data has been saved to 'chunked_data.json'")
