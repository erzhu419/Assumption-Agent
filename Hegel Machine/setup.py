"""Compatibility shim for older pip/setuptools frontends."""

from setuptools import find_packages, setup


setup(
    name="hegel-machine",
    version="0.85.0",
    description=(
        "Executable hypothesis discovery, active experimentation, "
        "and revisable cross-task knowledge."
    ),
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["cryptography>=41"],
    extras_require={"dev": ["pytest>=8"]},
    entry_points={"console_scripts": ["hegel-machine=hegel_machine.cli:main"]},
)
