"""Запуск и сохранение одного эпизода."""

__all__ = ["run_episode", "save_episode"]


def __getattr__(name):
    if name in __all__:
        from .loop import run_episode, save_episode
        return {"run_episode": run_episode, "save_episode": save_episode}[name]
    raise AttributeError(name)
