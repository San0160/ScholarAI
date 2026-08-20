from ai_research_assistant.citation.source_mapper import SourceMapper
from ai_research_assistant.entity.document import Document


documents = [

    Document(
        page_content="Adam optimizer...",
        metadata={
            "filename": "attention.pdf",
            "page": 7,
            "chunk_id": "attention.pdf_57"
        }
    ),

    Document(
        page_content="Training details...",
        metadata={
            "filename": "attention.pdf",
            "page": 8,
            "chunk_id": "attention.pdf_66"
        }
    ),

    Document(
        page_content="Transformer architecture...",
        metadata={
            "filename": "attention.pdf",
            "page": 2,
            "chunk_id": "attention.pdf_16"
        }
    )
]


mapper = SourceMapper()

sources = mapper.build(
    documents
)


print("=" * 80)
print("SOURCE MAPPING")
print("=" * 80)

for source_id, citation in sources.items():

    print(
        f"Source {source_id} → "
        f"{citation.document} | "
        f"Page {citation.page} | "
        f"Chunk {citation.chunk_id}"
    )