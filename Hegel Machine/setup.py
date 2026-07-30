"""Compatibility shim for older pip/setuptools frontends."""

from setuptools import find_packages, setup


setup(
    name="hegel-machine",
    version="0.1.0",
    description=(
        "A bounded, auditable kernel for structural-law recognition "
        "and conservative theory evolution."
    ),
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[],
    extras_require={"dev": ["pytest>=8"]},
    entry_points={"console_scripts": ["hegel-machine=hegel_machine.cli:main"]},
)
