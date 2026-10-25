"""lazyspotter's version, shown in the web app's header (bump it with every release in README's change log)."""
import subprocess
from pathlib import Path

VERSION = "2.6.0"


def full_version() -> str:
    """VERSION plus the git commit it runs from (and "+" with uncommitted changes), when this is a git checkout."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=Path(__file__).parent, capture_output=True, text=True,
                                        timeout=5).stdout.strip()
        commit = run("rev-parse", "--short", "HEAD")
        dirty = "+" if run("status", "--porcelain", "--untracked-files=no") else ""
        return f"{VERSION} ({commit}{dirty})" if commit else VERSION
    except (OSError, subprocess.SubprocessError):
        return VERSION
