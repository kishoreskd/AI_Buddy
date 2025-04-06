from fastapi import FastAPI
import torch
from models.response_model import QueryResult, QueryResponse, ErrorResponse, LLMResponse
from models.request_model import QueryRequest
from services.chroma_client import ChromaClient
from services.embedding_generator import EmbeddingGenerator
from transformers import pipeline
import re


# from langchain.llms import ollama

app = FastAPI()
chroma_client = ChromaClient()
embedding_model = EmbeddingGenerator()


@app.get("/", response_model=QueryResponse)
def health_check():
    return {"status": "ok", "results": []}


def generate(query, vectorResult):
    # Formatting inputs
    system_prompt = "You are a Q&A assistant. Your goal is to answer questions accurately based on the instructions and context provided."

    formatted_context = "\n".join([doc.chunk_text for doc in vectorResult])

    # Creating prompt
    prompt = f"{query}\n{formatted_context}\n"

    print(f"Formatted context: {formatted_context}")

    # Initializing pipeline
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Applying chat template
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    formatted_prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    formatted_prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generating response
    outputs = pipe(
        formatted_prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    response = outputs[0]["generated_text"]

    # Cleaning the response
    response = re.sub(r"<\|.*?\|>", "", response)  # Remove template tokens
    response = re.sub(r"</s>", "", response)  # Remove end-of-sequence tokens
    response = response.replace(system_prompt, "").strip()  # Remove system prompt

    return response


@app.post(
    "/query", response_model=LLMResponse, responses={500: {"model": ErrorResponse}}
)
def get_similar_responses(request: QueryRequest):
    try:
        embedding = embedding_model.get_embeddings(request.input)
        results = chroma_client.query_embedding(embedding, top_k=request.top_k)

        response = [
            QueryResult(
                match=idx + 1,
                page_number=metadata["page"],
                chunk_text=metadata["chunk_text"],
                similarity_score=distance,
                document_name=metadata["document_name"],
            )
            for idx, (metadata, distance) in enumerate(
                zip(results["metadatas"][0], results["distances"][0])
            )
        ]

        llm_esult = generate(request.input, response)

        llm_responses = LLMResponse(
            status="success",
            response=llm_esult,
            query=request.input,
            vector_results=response,
        )

        return llm_responses

    except Exception as e:
        raise RuntimeError(f"Error processing request: {e}")
