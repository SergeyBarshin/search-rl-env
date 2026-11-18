"""Чтение входов бенчмарка и независимая оценка сохранённых результатов."""

from collections import Counter, defaultdict
import csv
from statistics import mean

from search_rl_env.contracts.types import EpisodeResult, StopReason
from search_rl_env.search.acceptance import distribution

BENCHMARK_LANGUAGES = {"browsecomp-plus": "en", "ru-browsecomp-plus": "ru", "trust": "ru", "fixture": "en"}
RECALL_CUTOFF = 10


def read_questions(path):
    questions = {}
    with path.open(encoding="utf-8", newline="") as source:
        for number, fields in enumerate(csv.reader(source, delimiter="\t"), 1):
            if number == 1 and fields == ["query_id", "query"]:
                continue
            if len(fields) != 2:
                raise ValueError(f"Некорректная строка queries.tsv №{number}.")
            query_id, question = fields
            if not query_id.strip() or not question.strip() or query_id in questions:
                raise ValueError(f"Некорректная строка queries.tsv №{number}.")
            questions[query_id] = question
    if not questions:
        raise ValueError("Файл вопросов пуст.")
    return questions


def read_evidence(path):
    evidence = defaultdict(set)
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(f"Некорректная строка TREC qrels №{number}.")
            query_id, _, doc_id, relevance = fields
            if int(relevance) > 0:
                evidence[query_id].add(doc_id)
    return dict(evidence)


def _document_ids(ids, benchmark):
    if benchmark == "trust":
        return {identifier.split("_", 1)[0] for identifier in ids}
    return set(ids)


def score_retrieval(result: EpisodeResult, evidence: set[str], benchmark: str):
    if not evidence:
        raise ValueError("Для Recall нужен непустой набор положительных qrels.")
    observed = _document_ids(
        (hit["doc_id"] for step in result.trajectory for action in step.actions
         if action.name == "search" and action.result.get("ok") for hit in action.result["hits"]),
        benchmark,
    )
    selected = _document_ids(result.selected_document_ids[:RECALL_CUTOFF], benchmark)
    scorable = result.stop_reason != StopReason.INFRASTRUCTURE_ERROR
    return {
        "task_id": result.task_id, "stop_reason": result.stop_reason.value,
        "ranking_source": result.ranking_source,
        "recall_at_10": len(selected & evidence) / len(evidence) if scorable else None,
        "observed_evidence_recall": len(observed & evidence) / len(evidence) if scorable else None,
        "rounds": max((step.retrieval_round or 1 for step in result.trajectory), default=0),
        "search_calls": result.search_calls, "invalid_actions": result.invalid_actions,
        "duration_seconds": result.duration_seconds, "usage": result.usage.model_dump(mode="json"),
        "error_code": result.error.code if result.error else None,
    }


def summarize_retrieval(rows, requested):
    scored = [row for row in rows if row["recall_at_10"] is not None]
    costs = [row["usage"]["cost"] for row in rows]
    return {
        "requested": requested, "recorded": len(rows), "scored": len(scored),
        "coverage": len(scored) / requested,
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
        "ranking_sources": dict(Counter(row["ranking_source"] for row in rows if row.get("ranking_source"))),
        "recall_at_10": mean(row["recall_at_10"] for row in scored) if scored else None,
        "observed_evidence_recall": mean(row["observed_evidence_recall"] for row in scored) if scored else None,
        "latency": distribution([row["duration_seconds"] for row in rows]),
        "known_cost_usd": sum(value for value in costs if value is not None),
        "unknown_episode_costs": sum(value is None for value in costs),
    }
