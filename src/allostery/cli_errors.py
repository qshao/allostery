from __future__ import annotations

USER_ERROR = 1
USAGE_ERROR = 2
BACKEND_ERROR = 3


def _direct_code(exc: BaseException) -> int | None:
    # FileNotFoundError is an OSError subclass but is a user error, so check it first.
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return USER_ERROR
    if isinstance(exc, (ImportError, OSError)):
        return BACKEND_ERROR
    return None


def exit_code_for(exc: BaseException) -> int | None:
    code = _direct_code(exc)
    if code is not None:
        return code
    cause = exc.__cause__
    if cause is not None:
        return _direct_code(cause)
    return None


__all__ = ["BACKEND_ERROR", "USAGE_ERROR", "USER_ERROR", "exit_code_for"]
