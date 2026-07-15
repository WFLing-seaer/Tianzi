try:
    from core.api import API  # type: ignore  # noqa: F401

    IN_AHA = True
except ImportError, ModuleNotFoundError:
    IN_AHA = False


if IN_AHA:
    from .ahainit import *  # noqa: F403
