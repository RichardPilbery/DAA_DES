import textwrap
import warnings
import pytest

def fail_with_message(message: str):
    """Cleanly formatted pytest failure message."""
    pytest.fail(textwrap.dedent(message))

def warn_with_message(message: str):
    """Cleanly formatted warning message."""
    warnings.warn(textwrap.dedent(message), UserWarning)
