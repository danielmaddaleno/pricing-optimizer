"""Checks that the installed package declares what the code imports."""

from importlib.metadata import PackageNotFoundError, requires

import pytest

RUNTIME_REQUIREMENTS = ["pandas", "numpy", "scikit-learn", "scipy", "pydantic"]


def test_runtime_dependencies_are_declared():
    # Without these in pyproject, pip install . builds a package that imports
    # nothing, since requirements.txt is not part of the distribution.
    try:
        declared = requires("pricing-optimizer")
    except PackageNotFoundError:
        pytest.skip("package not installed, run pip install -e .")

    install_requires = [dep for dep in declared or [] if "extra ==" not in dep]
    names = [dep.split(">")[0].split("=")[0].strip() for dep in install_requires]
    assert sorted(names) == sorted(RUNTIME_REQUIREMENTS)
