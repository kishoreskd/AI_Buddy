from langchain.text_splitter import RecursiveCharacterTextSplitter

# from pdf_loader import PDFLoader
# import json


class TextChunker:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, extracted_data: list[dict]) -> list[dict]:
        chunked_data = []
        for data in extracted_data:
            chunks = self.splitter.split_text(data["text"])
            for chunk in chunks:
                chunked_data.append(
                    {
                        "chunk_text": chunk,
                        "page": data["page"],
                        "document_name": data["document_name"],
                    }
                )
        return chunked_data


# if __name__ == "__main__":
#     pdf_loader = PDFLoader("files/clinical_skills_guidance.pdf")
#     chunker = TextChunker()
#     chunked_datas = chunker.split_text(pdf_loader.extract_text())

#     with open("chunked_data.json", "w", encoding="utf-8") as json_file:
#         json.dump(chunked_datas, json_file, ensure_ascii=False, indent=4)

#     print("Chunked data has been saved to 'chunked_data.json'")
