import fitz
import json
import os


class PDFLoader:
    def __init__(self, filePath: str):
        self.filePath = filePath

    def extract_text(self) -> list[dict]:
        doc = fitz.open(self.filePath)
        extracted_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            # Clean the text by removing unwanted characters
            cleaned_text = text.replace("\n", " ").strip()
            extracted_data.append(
                {
                    "page": page_num + 1,
                    "text": cleaned_text,
                    "document_name": os.path.basename(self.filePath),
                }
            )

        # print(extracted_data)
        return extracted_data


# if __name__ == "__main__":
#     pdf_loader = PDFLoader("files/clinical_skills_guidance.pdf")

#     with open("pdf_extracted_data.json", "w", encoding="utf-8") as json_file:
#         json.dump(pdf_loader.extract_text(), json_file, ensure_ascii=False, indent=4)
