import importlib
import sys

INSTALL_HINT = (
    "nodestep sandbox needs the sandbox extra: "
    'uv add --dev "nodestep[sandbox] @ git+https://github.com/nodestep-ai/nodestep"'
)
EXTRA_MODULES = ("typer", "starlette", "jinja2", "uvicorn")


def missing_extra_modules() -> list[str]:
    """Return the modules of the ``sandbox`` extra that cannot be imported.

    Returns
    -------
    list[str]
    """
    missing: list[str] = []
    for name in EXTRA_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def main() -> None:
    """Run the ``nodestep`` command.

    Only the standard library is imported until the ``sandbox`` extra is known
    to be installed. Without it, the install hint goes to stderr and the
    process exits with code 2.
    """
    if missing_extra_modules():
        sys.stderr.write(INSTALL_HINT + "\n")
        raise SystemExit(2)
    from nodestep_sandbox.commands import app

    app()
