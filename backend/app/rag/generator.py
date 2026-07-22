"""
Turns retrieved chunks + a question into a grounded answer.

Two backends, both HuggingFace:
  1. Inference API (huggingface_hub.InferenceClient) if HF_API_TOKEN is
     set -- better quality instruction-tuned models, needs network.
  2. A small local flan-t5 pipeline (transformers) as a free, fully
     offline fallback so the project runs with zero API keys.

Swapping which backend is active is just an env var
(HF_API_TOKEN) -- no code changes needed, which is worth mentioning if
asked about deployment cost tradeoffs.
"""
from functools import lru_cache
from typing import List

from app.config import settings
from app.models.schemas import SourceChunk

NO_CONTEXT_ANSWER = (
    "I couldn't find anything in the uploaded documents that answers this. "
    "Try rephrasing, or upload a document that covers this topic."
)

SYSTEM_INSTRUCTIONS = (
    "You are a document assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, say you "
    "don't know -- never make up information. Keep the answer concise."
)


def _build_prompt(question: str, chunks: List[SourceChunk]) -> str:
    context = "\n\n".join(f"[Source {i+1} - {c.source}]\n{c.text}" for i, c in enumerate(chunks))
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


@lru_cache(maxsize=1)
def _get_local_pipeline():
    from transformers import pipeline
    return pipeline("text2text-generation", model=settings.local_generation_model)


def _generate_via_hf_api(prompt: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=settings.hf_api_token)
    response = client.chat_completion(
        model=settings.generation_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _generate_locally(prompt: str) -> str:
    pipe = _get_local_pipeline()
    # flan-t5 has a small context window -- keep the prompt tight.
    truncated = prompt[-3000:]
    output = pipe(truncated, max_new_tokens=256, do_sample=False)
    return output[0]["generated_text"].strip()


def generate_answer(question: str, chunks: List[SourceChunk]) -> str:
    if not chunks:
        return NO_CONTEXT_ANSWER

    prompt = _build_prompt(question, chunks)

    if settings.hf_api_token:
        try:
            return _generate_via_hf_api(prompt)
        except Exception as e:  # noqa: BLE001 - fall back rather than 500
            return _generate_locally(prompt) + f"\n\n(Note: hosted model call failed, used local fallback: {e})"

    return _generate_locally(prompt)
