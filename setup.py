from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

VERSION = "0.0.1"
PROJECT_NAME = "ScholarAI"
PACKAGE_NAME = "ai_research_assistant"

AUTHOR = "Sandeep"
AUTHOR_EMAIL = "your_email@example.com"  # Update this

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="An AI-powered Research Assistant using RAG and AI Agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-github-username/ScholarAI",  # Update later

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.10",

    install_requires=[],

    include_package_data=True,
    zip_safe=False,
)