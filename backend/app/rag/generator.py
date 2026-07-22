"""
Turns retrieved chunks + a question into a grounded answer.

Two backends, both HuggingFace:
  1. HF Inference API (router.huggingface.co) if HF_API_TOKEN is set.
     Uses the "hf-inference" provider which supports the free tier.
     Default model: HuggingFaceH4/zephyr-7b-beta (gated but free-tier
     accessible) — override with GENERATION_MODEL in .env.
     Note: the legacy InferenceClient().chat_completion() path against
     api-inference.huggingface.co was decommissioned in late 2025; we
     now pass provider="hf-inference" explicitly to hit the new router.
  2. A small local flan-t5 pipeline (transformers) as a free, fully
     offline fallback so the project runs with zero API keys.

Swapping which backend is active is just an env var (HF_API_TOKEN).
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
    """
    Uses the HF Inference API via the new router (router.huggingface.co).
    The legacy api-inference.huggingface.co endpoint was decommissioned
    in late 2025; passing provider="hf-inference" routes to the new stack.

    Free-tier note: large 7B+ instruction models (e.g. Mistral-7B) are
    not reliably available on the free hf-inference tier as of mid-2025.
    The default GENERATION_MODEL is set to HuggingFaceH4/zephyr-7b-beta
    which has better free-tier availability, but you can override it with
    any model your token has access to.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        provider="hf-inference",
        api_key=settings.hf_api_token,
    )
    response = client.chat.completions.create(
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
        except Exception as e:  # noqa: BLE001
            # Surface the real error rather than silently falling back and
            # appending a confusing note. If the hosted call fails, raise
            # so the caller can decide — pipeline.py catches and returns
            # the local result with a clear "hosted unavailable" message.
            raise RuntimeError(
                f"HF Inference API call failed: {e}. "
                "Check your HF_API_TOKEN and that GENERATION_MODEL is "
                "available on the free hf-inference tier."
            ) from e

    return _generate_locally(prompt)
