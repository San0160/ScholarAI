from dataclasses import dataclass
from ai_research_assistant.entity.document import Document


@dataclass
class RetrievalResult:
    document: Document
    score: float