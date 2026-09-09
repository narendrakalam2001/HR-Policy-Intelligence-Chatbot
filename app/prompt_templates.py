"""
app/prompt_templates.py

All prompt text lives here, isolated from chain/orchestration logic, so
prompts can be iterated on and reviewed independently (this file is what
you'd hand to a prompt-engineering reviewer).

Two templates are defined:
    1. SYSTEM_PROMPT       — persona + hard rules (grounding, citation,
                              refusal behaviour) for the HR policy assistant.
    2. build_human_prompt  — assembles the retrieved context, chat history,
                              and the user's question into the final
                              human-turn message sent to Gemini.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

SYSTEM_PROMPT = """\
You are the HR Policy Intelligence Assistant, an internal chatbot that \
answers employee questions about HR policy, labour law, and compliance \
topics (leave policy, gratuity, maternity benefit, POSH, shops & \
establishments regulations, and similar company/government policy \
documents).

Follow these rules strictly:

1. GROUNDING: Answer ONLY using the information in the "CONTEXT" section \
below. Do not use outside knowledge, and do not guess. If the context does \
not contain enough information to answer confidently, say so explicitly \
instead of speculating.

2. CITATIONS: Every factual claim must be traceable to the context. Reference \
which document and page you're drawing from naturally within your answer \
(e.g. "per the Gratuity Act, p.4..."). Do NOT add a separate "Sources:" \
summary line at the end of your answer — the application displays an \
accurate source list separately, generated from the actual retrieved \
documents rather than your own recall of them.

3. OUT OF SCOPE: If the question is unrelated to HR policy, labour law, or \
the documents provided (e.g. general chit-chat, coding help, current \
events), politely decline and redirect the user to ask an HR-policy-related \
question. Do not attempt to answer from general knowledge.

4. TONE: Be concise, professional, and neutral. This is a workplace tool — \
avoid casual language, humor, or personal opinions on policy matters.

5. UNCERTAINTY: If only partial information is available, answer with what \
is supported and clearly flag what is not covered by the provided \
documents. Never present an inference as a documented policy fact.

6. NO LEGAL ADVICE DISCLAIMER: You provide informational summaries of \
policy documents, not legal advice. If a question requires legal judgment \
on a specific personal situation, note that the user should confirm with \
HR or a qualified professional.
"""


def format_context(
    scored_chunks: Sequence[Tuple[str, str, int]],
) -> str:
    """
    Render retrieved chunks into a single context string the LLM can cite
    against.

    Args:
        scored_chunks: Sequence of (chunk_text, source_file, page_number)
            tuples, already ordered by relevance (most relevant first).

    Returns:
        A formatted context block, or a placeholder if nothing was retrieved.
    """
    if not scored_chunks:
        return "(No relevant context was retrieved from the knowledge base.)"

    blocks = []
    for i, (text, source_file, page_number) in enumerate(scored_chunks, start=1):
        blocks.append(f"[Excerpt {i} — {source_file}, p.{page_number}]\n{text}")
    return "\n\n".join(blocks)


def format_chat_history(history: Sequence[Tuple[str, str]]) -> str:
    """
    Render prior conversation turns into a compact transcript for context.

    Args:
        history: Sequence of (user_message, assistant_message) tuples,
            oldest first. Callers are responsible for truncating this to
            the desired memory window (e.g. last 5 turns) before calling.

    Returns:
        A formatted transcript string, or an empty-history placeholder.
    """
    if not history:
        return "(This is the start of the conversation.)"

    lines: List[str] = []
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


def build_human_prompt(
    question: str,
    context: str,
    chat_history: str,
) -> str:
    """
    Assemble the final human-turn message sent to the LLM.

    Args:
        question: The (preprocessed) current user question.
        context: Output of `format_context`.
        chat_history: Output of `format_chat_history`.

    Returns:
        The full human-turn prompt string.
    """
    return f"""\
PREVIOUS CONVERSATION (for follow-up context only — the CONTEXT section \
below is still the only source of factual truth):
{chat_history}

CONTEXT (retrieved policy document excerpts):
{context}

CURRENT QUESTION:
{question}

Answer the current question using only the CONTEXT above, following all \
system rules.\
"""


OUT_OF_SCOPE_FALLBACK = (
    "I couldn't find anything relevant to that in the HR policy and labour "
    "law documents I have access to. I'm best suited to questions about "
    "leave policy, gratuity, maternity benefit, POSH, and similar HR/"
    "compliance topics — could you rephrase your question around one of "
    "those areas, or check with your HR team directly for anything else?"
)