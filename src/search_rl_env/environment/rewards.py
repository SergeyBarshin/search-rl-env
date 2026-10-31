"""Детерминированная награда для первого retrieval-примера."""


def recall_at_k(document_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Для Recall нужно непустое множество эталонных документов.")
    selected = set(list(dict.fromkeys(document_ids))[:k])
    return len(selected & relevant) / len(relevant)
