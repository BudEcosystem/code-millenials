import os
import setuptools
from typing import List, Set

ROOT_DIR = os.path.dirname(__file__)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(os.path.join(ROOT_DIR, "requirements.txt")) as f:
        requirements = f.read().strip().split("\n")

    return requirements

setuptools.setup(
    name="code-millenials",
    version='0.0.1',
    author="Bud Ecosyste,",
    license="Apache 2.0",
    url="https://github.com/BudEcosystem/code-millenials",
    project_urls={
        "Homepage": "https://github.com/BudEcosystem/code-millenials"
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("assets")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    package_data={"code-millenials": ["py.typed"]},
)