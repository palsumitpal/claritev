"""Grounded answer generation; citations are tied to retrieved PDF page metadata."""
from openai import OpenAI
from .config import CHAT_MODEL

SYSTEM = """You are an assistant helping users navigate the 2026 CMS Medicare NCCI Policy Manual.
Answer only from the supplied excerpts. Do not invent policy, code rules, exceptions, or citations.
If the excerpts do not support an answer, say so and suggest what section or additional context is needed.
Distinguish what the manual explicitly says from your interpretation. This is informational support,
not coding, billing, legal, or reimbursement advice. Never imply that a citation proves more than
the cited excerpt. Cite every material policy claim using [PDF p. N] or [PDF pp. N-M]."""

def answer_question(question, hits, model=CHAT_MODEL):
    client = OpenAI()
    context = []
    for i, h in enumerate(hits, 1):
        context.append(
            f"[SOURCE {i}; PDF pages {h['page_start']}-{h['page_end']}; "
            f"section: {h.get('section','unknown')}]\n{h['text']}"
        )
    prompt = f"""Question: {question}

Retrieved excerpts:
{chr(10).join(context)}

Write a concise but useful answer. Cite page numbers inline in the form [PDF p. 12].
If sources conflict or are incomplete, explain the limitation."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""
