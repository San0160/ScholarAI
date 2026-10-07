# ScholarAI: future improvements (rough backlog)

Compiled 2026-10-06, after the first Railway deployment. Rough and unordered within each group. Items marked (yours) came from Sandy's own list.

## Suggested order

1. Evaluation, because it produces the number the resume is missing.
2. Heading handling in the PDF loader, because it loses content today.
3. Tests and pinned dependencies, so later changes are safe.
4. Public-demo protections (rate limits, per-visitor isolation).
5. Paid-plan features (chat history, large-document summaries).

## Evaluation (important) (yours)

- Build a small gold set: 20 to 30 questions over 2 or 3 documents, each with the expected answer and the page it comes from.
- Retrieval metrics: hit rate at k and MRR, measured with and without the reranker. The difference is a concrete resume number.
- Citation accuracy: does the cited page match the gold page.
- Answer faithfulness: an LLM-judged check that each answer is supported by the retrieved text. Needs pacing on the free plan (8K tokens per minute).
- Use the same set to settle open questions: chunk size 350 vs the embedding model's maximum input length (possible silent truncation, never confirmed), top_k, paragraph-aware chunking on vs off.

## Ingestion and chunking

- (yours) Headings are removed from the text: stored lowercased in `section` metadata, which nothing uses. Keep the heading as the first line of its section, or add `Section:` to the context header.
- (yours) Any short bold line starting with a capital counts as a heading and is dropped, so short bold body text can be lost.
- Filter near-empty chunks (a lone ID number, a bare numbered list) and standalone page numbers in body text.
- Scanned or image-only PDFs are unsupported; OCR would be needed.
- Confirm `start_char` / `end_char` are present on chunk metadata. They are offsets into the cleaned text, not the original file.
- Status line says "16 document(s), 21 chunk(s)"; show pages instead, or just "Ready".

## Retrieval and answers

- Page-range questions ("the first 2 pages") are not enforced by the pipeline; detect the range and filter by page.
- Metadata filters run after retrieval, so they can return nothing on a large index. Filter before the search.
- Whole-document detection is a keyword pattern; replace with an explicit mode or an LLM-based router.
- Per-point clickable page markers inside the answer, in place of only the citations card.
- Remove `CitationMatcher` from the generation path: nothing uses its output and it re-embeds every chunk on every question.
- Move citation settings and other hardcoded numbers into config.
- Hybrid search (keyword plus vector) for exact terms and names.
- Render full Markdown in answers (currently bold and line breaks only), and stream answers as they generate.

## Paid-plan features

- (yours) Groq paid plan, then chat history, memory and follow-up questions.
- Summaries of large documents in several passes (map-reduce), which the free plan's per-minute limit rules out.
- Raise `max_context_tokens` beyond 6500 once the limit allows.

## Public deployment

- All visitors share one index and one Groq quota. Add per-visitor isolation and rate limiting.
- Show a clear "rate limit reached, try again in a minute" message instead of "Answer generation failed".
- Uploaded files stay on the server until restart; delete them after indexing or after a time limit.
- The index is wiped on every start by design. A Railway volume would allow persistence if wanted.
- Verify what `rebuild=True` does to the index (replace one file's chunks, or duplicate them). `IndexingPipeline.run()` was never fully reviewed.
- A failed re-index deletes the previously working file.
- `/health` reports ok even if a pipeline failed to load; make it a readiness check.
- Serve PDF.js from `static/` instead of a CDN.
- Watch Railway usage against the $5 included; consider Serverless.

## Engineering quality

- (yours) Make use of `params.yaml`: keep paths and providers in `config.yaml`, move tunable numbers (chunk size, overlap, top_k, candidate_k, token budgets, upload size limit) to `params.yaml`.
- Tests: none exist. Start with the chunker, text cleaner, citation utilities and context builder, and run them in CI.
- Pin dependency versions from `pip freeze`.
- Remove dead code (`EnvConfig`, unused formatter and matcher paths).
- Per-request logging of stage timings and token usage.
- Docker: run as a non-root user.
- DOCX preview; highlight the cited passage in text previews.

- README with an architecture diagram, a short demo recording and the live link.