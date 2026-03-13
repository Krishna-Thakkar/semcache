"""rag_usage.py — Question extraction for RAG-style prompts.

In RAG pipelines, prompts carry a context block followed by a question.
Enabling extract_question=True strips the context so that two prompts
asking the same question with different context still hit the same cache
entry.

Run with:
    python examples/rag_usage.py
"""

from semcache import SemCache

cache = SemCache(extract_question=True)


def fake_llm(prompt: str) -> str:
    print("  [LLM called]")
    return f"Answer: {prompt}"


rag_prompt_1 = """\
Context:
A convolutional neural network (CNN) is a class of deep learning model
commonly applied to image recognition tasks.

Question: What is CNN?
"""

rag_prompt_2 = """\
Context:
Different background text about computer vision research from 2024.

Question: What is CNN?
"""

print("First RAG prompt (different context, same question):")
print(cache.ask(rag_prompt_1, fake_llm))

print("\nSecond RAG prompt (different context, same question — cache hit):")
print(cache.ask(rag_prompt_2, fake_llm))

print("\nStats:", cache.stats())
