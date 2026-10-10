# RAG-Astra - System Architecture

## 1. Purpose

RAG-Astra is an enterprise AI Knowledge & Decision System that transforms organizational documents into searchable, evidence-grounded answers.

This document describes the intended architecture for V1. It does not imply that the components have been implemented.

## 2. High-level architecture

The system consists of six primary layers:

1. **Frontend:** User interface for knowledge bases, document management, search, chat, and citations.
2. **API layer:** Authentication, request validation, business logic, and application endpoints.
3. **Ingestion pipeline:** File validation, text extraction, normalization, chunking, embedding generation, and indexing.
4. **Retrieval and RAG layer:** Query processing, hybrid retrieval, context assembly, answer generation, and citation mapping.
5. **Data and background processing:** PostgreSQL, pgvector, Redis, and background workers.
6. **Evaluation and observability:** Automated tests, retrieval-quality evaluation, structured logging, metrics, and error tracking.

## 3. Technology stack

| Component          | Technology                          | Responsibility                          |
| ------------------ | ----------------------------------- | --------------------------------------- |
| Frontend           | Next.js, TypeScript, Tailwind CSS   | User experience                         |
| Backend            | Python, FastAPI, Pydantic           | API and application logic               |
| Database           | PostgreSQL                          | Persistent application data             |
| Vector search      | pgvector                            | Embedding similarity search             |
| ORM and migrations | SQLAlchemy, Alembic                 | Database access and schema evolution    |
| Background jobs    | Redis and a task worker             | Asynchronous ingestion                  |
| AI providers       | Configurable embedding and LLM APIs | Embeddings and answer generation        |
| Infrastructure     | Docker, Docker Compose              | Reproducible development and deployment |
| Testing            | pytest and frontend test tooling    | Correctness and regression prevention   |

## 4. Document ingestion flow

1. An authenticated user selects an authorized knowledge base and uploads a document.
2. The API validates the file type, size, and request.
3. Document metadata and processing status are recorded.
4. A background job extracts text from the document.
5. Extracted content is normalized and split into chunks.
6. Each chunk is associated with its document and version.
7. Embeddings are generated and stored with the corresponding chunk records.
8. Search indexes are updated and processing status is reported.
9. Failures are recorded and handled through controlled retries.

Supported initial formats: PDF, DOCX, TXT, and Markdown.

## 5. Retrieval and answer-generation flow

1. The user submits a question.
2. The API authenticates the user and resolves authorized knowledge-base scope.
3. The retrieval pipeline normalizes the query and optionally rewrites or decomposes it.
4. Semantic and keyword retrieval identify candidate passages.
5. Results are filtered by access permissions and relevant metadata.
6. Candidate results are fused and optionally reranked.
7. Selected passages are assembled into bounded context.
8. The language model generates an answer grounded in that context.
9. Citations are mapped to the source document and passage.
10. Grounding checks determine whether the answer is sufficiently supported.
11. The system returns the answer, citations, and relevant metadata, or abstains when evidence is insufficient.

## 6. Data model principles

The initial data model is expected to include:

- **User:** Identity and account information.
- **Organization or tenant:** Logical security boundary for enterprise data.
- **Membership and role:** User permissions within an organization.
- **Knowledge base:** A managed collection of documents.
- **Document:** Logical identity and metadata of an uploaded file.
- **Document version:** A specific revision and its processing state.
- **Chunk:** A segment of extracted text linked to a document version.
- **Embedding:** Vector representation associated with a chunk.
- **Ingestion job:** Processing status, attempts, and errors.

The final schema will be refined before migrations are created. PostgreSQL is the source of truth for application records.

## 7. Security requirements

- Store secrets outside source control.
- Validate uploaded files and enforce size limits.
- Authenticate requests and authorize every protected operation.
- Enforce tenant and knowledge-base access controls in backend queries.
- Apply access filters before returning retrieved evidence to generation.
- Treat document contents as untrusted data, not as system instructions.
- Avoid exposing private content in logs or error responses.
- Test unauthorized access and cross-tenant isolation.
- Use secure transport and appropriate secret management in deployed environments.

## 8. Reliability and observability

- Use explicit document-processing states.
- Make ingestion jobs retryable and safe against duplicate execution.
- Use database transactions for consistent state changes.
- Log errors and processing outcomes with correlation identifiers.
- Measure API latency, ingestion duration, retrieval quality, and generation latency.
- Maintain regression tests and versioned evaluation datasets.
- Provide health checks for the API and required dependencies.

## 9. Architectural boundaries

V1 will begin as a modular monolith with clear boundaries between API endpoints, services, repositories, ingestion, retrieval, and RAG orchestration.

We will use PostgreSQL with pgvector before introducing a separate vector database. We will use background workers for long-running ingestion rather than blocking API requests.

Kubernetes, Kafka, graph RAG, and multi-agent orchestration are out of scope unless a demonstrated requirement justifies them.

## 10. Implementation roadmap

1. Repository foundation and architecture documentation.
2. Development environment and backend skeleton.
3. Configuration, database connectivity, and migrations.
4. Knowledge-base and document-management APIs.
5. Ingestion pipeline and background processing.
6. Embeddings, hybrid retrieval, and ranking.
7. Evidence-grounded generation and citations.
8. Evaluation, authorization, and security testing.
9. Frontend implementation and integration.
10. Deployment, monitoring, and performance validation.

## 11. Architecture status

**Status:** Proposed architecture.

Each capability will be marked implemented only after the corresponding code, tests, and verification are complete.
