# Medicare NCCI 2026 Manual Assistant application – RAG CLI based application

A RAG prototype for asking questions about the **2026 CMS Medicare National Correct Coding Initiative (NCCI) Policy Manual**.
The application expects PDF to have been downloaded locally

## What does it do
- Extracts page text from the local PDF.
- Builds two chunk collections: fixed-window which is the baseline and structure-aware chunks.
- Leverages vector, BM25 keyword, and hybrid (reciprocal-rank fusion) based retrieval.
- Uses OpenAI embeddings and an OpenAI chat model to synthesize answers from retrieved chunks.
- Displays page citations and retrieved source pages.

## Requirements
- Python 3.11 or newer
- An OpenAI API key with API access and billing configured
- The 2026 CMS NCCI PDF saved locally
- Internet access during ingestion and Q&A for OpenAI API calls

## Ingest the PDF
```bash
python -m src.indexer --pdf "PathTo-The-2026-ncci-medicare-policy-manual-all-chapters.pdf" --strategy all
```

`all` option creates Chroma collections and JSONL corpora for `fixed` and `structure`. 
Ingestion calls embedding API


## Ask questions
```bash
python -m src.chat --strategy structure --retrieval hybrid
```

Other options for asking questions:

```bash
python -m src.chat --strategy fixed --retrieval vector
python -m src.chat --strategy structure --retrieval vector
python -m src.chat --strategy structure --retrieval bm25
python -m src.chat --strategy structure --retrieval hybrid
```

## Architecture

### Ingestion control flow
1. Open the PDF and extract text page-by-page.
2. Preserve the original PDF page index.
3. Create chunks using the selected strategy.
4. Batch chunks to the embedding endpoint.
5. Store vectors, text, and metadata in Chroma.
6. Store a JSONL corpus for BM25 retrieval.

### Query control flow
1. Read a question from the CLI.
2. Embed the question for vector retrieval (if selected).
3. Run BM25 over the local text corpus (if selected).
4. For hybrid mode, combine ranked lists with reciprocal-rank fusion (RRF).
5. Send the top passages, page ranges, and section labels to the chat model.
6. Ask the model to answer only from those passages and cite page numbers.
7. Print the answer and retrieved source list.

## Limitations and responsible use

- This is a prototype only.
- Does not handle tables, footnotes, headers/footers, or complex PDF layout.
- Retrieval can fail even when the answer exists in the PDF.
- API usage incurs costs and is subject to OpenAI account terms and data controls.
