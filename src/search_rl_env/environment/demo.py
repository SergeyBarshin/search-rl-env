"""Связанный синтетический пример SQL + RAG, без внешних данных."""

from search_rl_env.contracts import AgentTask, SearchHit

HYBRID_QUESTION = "Какие заказы подходят под компенсацию согласно регламенту?"
HYBRID_HITS = [
    SearchHit(doc_id="policy-001", rank=1, snippet=(
        "Регламент компенсаций v1: компенсация положена заказам региона RU "
        "со статусом delivered и задержкой доставки строго более 3 дней.")),
    SearchHit(doc_id="policy-002", rank=2, snippet=(
        "Статусы заказов: cancelled — отменён, pending — ожидает доставки. "
        "Такие заказы не участвуют в программе компенсаций.")),
]
# Эталон используется только оценщиком и тестами, не включается в промпт.
HYBRID_REFERENCE = ["ORD-101", "ORD-105"]


def hybrid_task(*, task_id: str = "hybrid-demo", question: str = HYBRID_QUESTION) -> AgentTask:
    return AgentTask(task_id=task_id, question=question, language="ru",
                     corpus_id="synthetic-compensation", corpus_revision="v1")
