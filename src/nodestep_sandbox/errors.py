class SandboxError(Exception):
    """Base class of the errors the sandbox reports to the user."""


class TargetError(SandboxError):
    """A graph or context target that cannot be loaded.

    Parameters
    ----------
    message : str
    details : str, optional
        The traceback of the loaded code, when importing or calling it raised.
    """

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.details = details


class FormError(SandboxError):
    """Submitted form values that cannot become run input or resume answers.

    Parameters
    ----------
    errors : dict[str, str]
        Message per form control name.
    """

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__(
            "; ".join(f"{name}: {message}" for name, message in errors.items())
        )
        self.errors = errors


class RunStateError(SandboxError):
    """A resume of a run that is not waiting for answers."""
