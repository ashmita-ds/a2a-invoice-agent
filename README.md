# A2A Invoice Action Agent (starter implementation)

A FastAPI implementation of the A2A HTTP+JSON surface for the invoice-action assignment.

## Important

This is a strong starter implementation, not a verified full-marks submission. The grader's exact schemas, receipt race behavior, persistent disk availability, and hidden audit cases must be tested against the assignment environment. Do not spend a limited Check until the local tests and protocol review are complete.

## 1. Local setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables. Use a long random token and an AI provider's API key.

PowerShell:
```powershell
$env:AGENT_TOKEN = "replace-with-a-long-random-secret"
$env:AI_API_KEY = "your-provider-api-key"
$env:AI_BASE_URL = "https://api.groq.com/openai/v1"
$env:AI_MODEL = "llama-3.1-8b-instant"
$env:DB_PATH = "./data/invoice_agent.sqlite3"
uvicorn main:app --reload --port 8000
```

Linux/macOS:
```bash
export AGENT_TOKEN='replace-with-a-long-random-secret'
export AI_API_KEY='your-provider-api-key'
export DB_PATH='./data/invoice_agent.sqlite3'
uvicorn main:app --reload --port 8000
```

Test:
- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/.well-known/agent-card.json`

## 2. Required headers

For every protected route:
```
Authorization: Bearer <exact AGENT_TOKEN>
A2A-Version: 1.0
Content-Type: application/a2a+json
```

The Agent Card is public. For the grader, submit the deployed URL ending in `/a2a/` (unless you change `BASE_PATH`).

## 3. AI provider

The app expects an OpenAI-compatible `/chat/completions` API. Do not commit API keys. Configure them in the host's environment settings.

## 4. Deployment

Deploy the repository to a Python web host that supports a persistent disk. The included `render.yaml` is a starting point, but Render's free web services may not provide persistent disk; if the instance restarts, SQLite state may be lost. Use a persistent disk or another durable SQLite/Postgres configuration. Confirm that the host doesn't redirect the submitted URL.

## 5. Before spending a Check

The assignment is unusually strict. Validate all of the following locally:
- Exact Agent Card and `supportedInterfaces` base URL.
- All POST requests require version and exact media type.
- Auth and user isolation on all protected endpoints.
- Same `(principal, messageId)` + same canonical message returns identical complete Task.
- Same message ID with changed message returns 409 `IDEMPOTENCY_CONFLICT`.
- Concurrent equivalent initial requests cannot make duplicate AI calls or create two tasks.
- Proposal cache is keyed by canonical package content and not batch/task IDs.
- Each proposal uses exactly the three decisive references from the action-determining paragraph.
- Results must be bound to stored task/context/batch/proposal/action and receipt nonce.
- Cancel/result race is atomic across workers and persistent storage.
- Terminal tasks never change.
- Responses stay under 512 KiB.
- Check and Save should each exercise a fresh audit receipt lifecycle.

## Known hardening still recommended

Before grading, improve:
1. Cross-worker atomic deduplication around AI inference (the current implementation has a small inference race).
2. Strict validation of message IDs and accepted output modes.
3. Receipt/cancel race semantics: currently SQLite transaction locking serializes writes, but cancellation/result routes should be exercised concurrently.
4. Add test cases with representative synthetic packages and inspect evidence-reference extraction.
5. Ensure a durable database is mounted at `DB_PATH`.
