# ScholarLink

A free, local clone of the core ScholarLink idea: search for research-fit professors,
generate personalized draft outreach emails, review them, and push them into your
actual Gmail Drafts folder. Nothing is ever auto-sent — you always send from Gmail
yourself.

Cost: $0. No paid API keys required.

## What it uses

- **Flask** — local web app, runs at `http://localhost:5000`
- **SQLite** — local database (`scholarlink.db`), created automatically on first run
- **Semantic Scholar API** — free, official, no key needed for light use — for
  finding professors and their research areas (Google Scholar has no official API
  and scraping it gets you blocked, so this is the practical free alternative)
- **LM Studio (local LLM)** — generates the draft text on your own machine, free,
  no per-token cost, works offline
- **Gmail API** — creates real drafts in your Gmail account (`gmail.compose` scope
  only — it can create/edit drafts, it cannot read your inbox or send mail)

## One-time setup

### 1. Python environment

```bash
cd scholar-link
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. LM Studio (for draft generation)

1. Open LM Studio, load a model (your Gemma setup works fine).
2. Go to the "Local Server" tab and start the server on the default port
   (`http://localhost:1234`).
3. Leave it running while you use ScholarLink.

If you'd rather use a different local server (Ollama, etc.), just edit the
`LM_STUDIO_URL` in `llm.py` to match — most local LLM servers speak the same
OpenAI-compatible chat completions format.

### 3. Gmail API (for pushing drafts into Gmail)

1. Go to https://console.cloud.google.com/ and create a new project.
2. In "APIs & Services" → "Library", enable the **Gmail API**.
3. In "APIs & Services" → "OAuth consent screen":
   - User type: External
   - Publishing status: leave it in **Testing** (this is fine forever for
     personal use — Google only requires verification for apps used by the
     public, and this app only has one user: you)
   - Add your own Google account under "Test users"
4. In "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID":
   - Application type: **Desktop app**
   - Download the resulting JSON
5. Save that file as `credentials/client_secret.json` in this project folder.

The first time you click "Create draft in Gmail" in the app, a browser window
will open asking you to log in and approve access. After that, a `token.json`
is cached in `credentials/` so you won't be asked again.

### 4. Run it

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Using it

1. **Profile** — fill in your name, background, interests, and (optionally)
   tweak the email template. Save.
2. **Find Professors** — search by topic + school, e.g. "controls robotics UC
   San Diego". Semantic Scholar doesn't have a strict school/department filter
   the way a curated directory would, so mixing topic + school keywords in one
   query gives the best results.
3. Click **Prepare draft** on a result — this pulls their recent papers, feeds
   your profile + their research into your local LLM, and drops a generated
   draft into your queue.
4. **Draft Queue** → **Review** — read it, edit the subject/body freely.
5. Semantic Scholar doesn't expose professor email addresses (neither does
   Google Scholar's public data, for privacy reasons) — you'll need to find
   the address yourself, usually one Google search away on the university's
   faculty directory page, and paste it in before pushing to Gmail. This is
   also the case in the original ScholarLink product for the same reason.
6. **Create draft in Gmail** — this creates the draft in your real Gmail
   account. Go to Gmail, review it one more time, and hit send yourself.

## Honest limitations vs. the commercial version

- No curated faculty directory — search quality depends on what's indexed
  in Semantic Scholar and how you phrase the query.
- No professor email addresses provided automatically (a legal/privacy line
  most tools in this space don't cross either).
- Single-user, local-only, no accounts or billing — by design, since this is
  for your own use.
- No credit system — draft generation is free and effectively unlimited
  since it's your own LLM doing the work, not a paid API.

## Extending it

Ideas if you want to keep building:
- Add a manual "edit professor email" field on the draft detail page instead
  of only via the database.
- Add a second search source (e.g. a university's public directory page,
  scraped with `requests` + `BeautifulSoup`, department by department) to fill
  gaps Semantic Scholar misses.
- Swap the SQLite email-template system for multiple named templates (e.g.
  one for CS professors, one for mechanical engineering).
