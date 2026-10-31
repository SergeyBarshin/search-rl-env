"""Запуск и сохранение одного эпизода."""

__all__ = ["run_episode", "save_episode", "RolloutRunner", "RolloutRequest"]


def __getattr__(name):
    if name in {'RolloutRunner', 'RolloutRequest'}:
        from .runner import RolloutRunner, RolloutRequest
        return {'RolloutRunner': RolloutRunner, 'RolloutRequest': RolloutRequest}[name]
    if name in {'run_episode', 'save_episode'}:
        from .loop import run_episode, save_episode
        return {"run_episode": run_episode, "save_episode": save_episode}[name]
    raise AttributeError(name)
