from setuptools import setup, find_packages


def get_requirements(file_path: str) -> list[str]:
    """Read dependencies from a requirements file, dropping the
    editable self-install line so it isn't listed as its own dependency."""
    with open(file_path, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip()]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

VERSION = "0.0.1"
PROJECT_NAME = "ScholarAI"
PACKAGE_NAME = "ai_research_assistant"

AUTHOR = "Sandeep"
AUTHOR_EMAIL = "sandeepaflc2000@gmail.com"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="An AI-powered research assistant using retrieval-augmented generation (RAG).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/San0160/ScholarAI",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.10",
    install_requires=get_requirements("requirements.txt"),

    include_package_data=True,
    zip_safe=False,

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)