"""Среда и харнесс поискового агента."""

__all__ = ["SearchTaskset", "SearchHarness"]


def __getattr__(name):
    if name == "SearchTaskset":
        from .environment.taskset import SearchTaskset
        return SearchTaskset
    if name == "SearchHarness":
        from .agent.harness import SearchHarness
        return SearchHarness
    raise AttributeError(name)
