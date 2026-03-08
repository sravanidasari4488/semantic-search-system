"""Data loader for 20 Newsgroups dataset."""

import re
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = "data/20_newsgroups/20_newsgroups"


def _strip_email_headers(text: str) -> str:
    """Remove email headers (everything before the first blank line).

    Usenet posts use RFC 822 email format. Lines like From:, Subject:,
    Organization:, Lines: contain metadata, not the actual message content.
    The blank line separates headers from body. Keeping headers would pollute
    embeddings with sender info and duplicate structural text across documents.
    """
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else text


def _strip_quoted_replies(text: str) -> str:
    """Remove quoted reply lines (lines starting with '>').

    In email/Usenet replies, quoted text from the original post is prefixed
    with '>'. This adds noise and often duplicates content from other documents,
    which hurts semantic search quality. We keep only the author's own text.
    """
    lines = [line for line in text.split("\n") if not line.strip().startswith(">")]
    return "\n".join(lines)


def _strip_signatures(text: str) -> str:
    """Remove email signatures (common patterns like '--').

    Signatures usually start with '-- ' on its own line (standard email/Usenet
    delimiter). They contain names, disclaimers, contact info—not useful for
    search. Removing them keeps embeddings focused on the actual message.
    """
    match = re.search(r"\n\s*--\s*\n", text)
    if match:
        return text[: match.start()].rstrip()
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse extra whitespace and trim.

    Excessive newlines and spaces add no meaning but increase token count and
    can dilute embeddings. We collapse 3+ newlines to 2 and strip edges.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess(text: str, **kwargs) -> str:
    """Preprocess raw newsgroup text for semantic search.

    Pipeline: headers -> quoted replies -> signatures -> whitespace.

    Args:
        text: Raw file content.
        **kwargs: Unused, for extensibility.

    Returns:
        Cleaned text.
    """
    if not text or not text.strip():
        return ""
    result = _strip_email_headers(text)
    result = _strip_quoted_replies(result)
    result = _strip_signatures(result)
    result = _normalize_whitespace(result)
    return result


def load_documents(source: str = DEFAULT_SOURCE, **kwargs) -> pd.DataFrame:
    """Load 20 Newsgroups dataset from local folder.

    Expects structure:
        source/
            category1/
                file1
                file2
            category2/
                ...

    Args:
        source: Path to dataset root (e.g., data/20_newsgroups/20_newsgroups).
        **kwargs: Unused, for API compatibility.

    Returns:
        DataFrame with columns: document_id, category, text.
    """
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {source_path}")

    rows = []

    for category_dir in source_path.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for file_path in category_dir.iterdir():
            if not file_path.is_file():
                continue

            document_id = file_path.name

            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw_text = ""

            text = preprocess(raw_text)

            if not text:
                continue

            rows.append(
                {
                    "document_id": document_id,
                    "category": category,
                    "text": text,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df[["document_id", "category", "text"]].copy()
