# PetTron 🤖

> **A daily AI agent that runs entirely on GitHub Actions — 100% free, zero servers.**

PetTron wakes up once a day, audits your GitHub repositories for bugs and security issues, opens Pull Requests with verified fixes, and scouts GitHub for freelance opportunities — all while you sleep.

---

## What it does

| Part | Task | Output |
|---|---|---|
| **A — Repo Auditor** | Scans your repos for security flaws, bugs, missing error handling | A verified Pull Request, ready for your review |
| **B — Opportunity Scout** | Searches GitHub Issues for freelance/bounty/help-wanted posts | 3 tailored reply drafts in your private digest |
| **Notifier** | Creates a daily digest Issue in your private log repo | GitHub Mobile push notification |

---

## Architecture

```
GitHub Actions (cron 6AM UTC daily)
         │
         ▼
   src/main.py (orchestrator)
    ┌────┴─────────────────────────────┐
    │                                  │
    ▼                                  ▼
PART A                             PART B
repo_auditor.py                    opportunity_scout.py
    │  GitHub API → fetch files         │  GitHub Search API
    ▼                                   ▼
llm_client.py                      llm_client.py
    │  Gemini 2.0 Flash                 │  Rank top 3, draft replies
    │  (Groq fallback)                  │
    ▼                                   │
verifier.py                            │
    │  Clone → patch → pytest/lint      │
    ▼                                   │
pr_manager.py                          │
    │  Open PR via GitHub API           │
    └──────────────┬────────────────────┘
                   ▼
             notifier.py
         Create Issue in pettron-logs
         → GitHub Mobile notification
```

---

## Setup Guide

### Prerequisites
- A GitHub account
- Python 3.12+ (for local testing only)
- Free API keys: Google AI Studio + GroqCloud

### Step 1 — Fork / Create Repos

1. Create the **bot repo** (this one): make it **PUBLIC** for unlimited free Actions minutes.
2. Create a **private log repo** (e.g. `YOUR_USERNAME/pettron-logs`) — set to **Private**.

### Step 2 — Get Your Free API Keys

| Service | URL | Time |
|---|---|---|
| Gemini API (Primary LLM) | [aistudio.google.com](https://aistudio.google.com) → Get API Key | ~1 min |
| Groq API (Fallback LLM) | [console.groq.com](https://console.groq.com) → API Keys | ~2 min |

### Step 3 — Create GitHub Personal Access Token

1. GitHub → **Settings** → **Developer settings** → **Tokens (classic)** → **Generate new token**
2. Name: `PetTron Bot Token`
3. Expiration: `No expiration`
4. Tick these scopes: ✅ `repo` ✅ `workflow` ✅ `read:discussion`
5. Copy the token (shown once only)

### Step 4 — Add GitHub Secrets

Go to: `PetTron repo → Settings → Secrets and variables → Actions`

| Secret Name | Value |
|---|---|
| `GH_PAT` | Your Personal Access Token |
| `GEMINI_API_KEY` | From Google AI Studio |
| `GROQ_API_KEY` | From GroqCloud |
| `PRIVATE_REPO_FOR_LOG` | `YOUR_USERNAME/pettron-logs` |

### Step 5 — Configure Your Target Repos

Edit `config/repos.yaml` and update the repo names to match your actual GitHub username and repos:

```yaml
repositories:
  - name: "YOUR_USERNAME/YourRepoName"
    branch: "main"
    language: "python"
    verify_cmd: "python -m compileall . -q"
    last_checked: null
```

### Step 6 — First Run

Push all files to GitHub, then:

```
PetTron repo → Actions tab → "PetTron Daily Agent" → Run workflow → Run workflow
```

Check your `pettron-logs` repo — a new Issue should appear within ~5 minutes.

---

## Running Tests Locally

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Free Tier Limits (why this stays free forever)

| Resource | Limit | PetTron's daily usage |
|---|---|---|
| GitHub Actions (public repo) | ♾️ Unlimited | ~3-5 minutes/day |
| Gemini 2.0 Flash | 1,500 req/day, 1M TPM | ~4 calls/day |
| Groq Llama 3.1 8B | 14,400 req/day | Fallback only |
| GitHub Search API | 30 req/min | ~10 calls/day |

---

## Safety Rules

- 🚫 **Never** auto-merges Pull Requests
- 🚫 **Never** auto-posts replies on public issues
- 🚫 **Never** touches `.github/workflows/` or `.env` files
- ✅ All patches verified by running the actual test/lint command before a PR is opened
- ✅ Self-correction loop: if a patch breaks tests, the LLM gets one chance to fix it; if it fails again, no PR is opened

---

## Project Structure

```
PetTron/
├── .github/workflows/daily_agent.yml   # Cron job + keep-alive commit
├── config/
│   ├── repos.yaml                      # Your target repos & verify commands
│   └── profile.json                    # Your skills, projects, pricing
├── src/
│   ├── main.py                         # Orchestrator entry point
│   ├── llm_client.py                   # Gemini + Groq with fallback
│   ├── schemas.py                      # Pydantic data models
│   ├── repo_auditor.py                 # Repo rotation + file fetching + LLM analysis
│   ├── verifier.py                     # Safe patching + verification gate
│   ├── pr_manager.py                   # Branch + commit + PR creation
│   ├── opportunity_scout.py            # GitHub Search + LLM ranking + reply drafts
│   └── notifier.py                     # Daily digest GitHub Issue
├── state/run_state.json                # Rotation state (auto-updated daily)
├── tests/test_patcher.py               # Unit tests for patching logic
└── requirements.txt
```

---

*Built with ❤️ by Junaid Mohi-Ud-Din*
