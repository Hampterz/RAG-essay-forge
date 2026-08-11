import sqlite3

DB_PATH = "scholarlink.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT,
            background TEXT,
            interests TEXT,
            resume_text TEXT,
            email_template TEXT,
            gemini_api_key TEXT,
            gemini_model TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS professors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            affiliation TEXT,
            department TEXT,
            research_interests TEXT,
            email TEXT,
            semantic_scholar_id TEXT UNIQUE,
            contacted INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            professor_id INTEGER,
            subject TEXT,
            body TEXT,
            status TEXT DEFAULT 'pending',
            gmail_draft_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (professor_id) REFERENCES professors (id)
        )
    """)

    # ensure a single profile row exists
    c.execute("SELECT COUNT(*) as cnt FROM profile")
    if c.fetchone()["cnt"] == 0:
        c.execute("""
            INSERT INTO profile (id, name, background, interests, resume_text, email_template, gemini_api_key, gemini_model)
            VALUES (1, '', '', '', '', ?, '', '')
        """, (DEFAULT_TEMPLATE,))

    conn.commit()
    conn.close()


DEFAULT_TEMPLATE = """Subject: Interest in your research on {research_topic}

Dear Professor {last_name},

My name is {student_name}, and I am {background}. I came across your work on \
{research_topic} and found it especially compelling because {personal_connection}.

{interests}

I would be grateful for the chance to learn more about your research, and to ask \
whether there might be any opportunities to contribute, even in a small capacity, \
to your lab's work.

Thank you for your time and consideration.

Best regards,
{student_name}
"""


def get_profile():
    conn = get_conn()
    row = conn.execute("SELECT * FROM profile WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else None


def save_profile(name, background, interests, resume_text, email_template, gemini_api_key, gemini_model):
    conn = get_conn()
    conn.execute("""
        UPDATE profile SET name=?, background=?, interests=?, resume_text=?, email_template=?, gemini_api_key=?, gemini_model=?
        WHERE id = 1
    """, (name, background, interests, resume_text, email_template, gemini_api_key, gemini_model))
    conn.commit()
    conn.close()


def upsert_professor(name, affiliation, department, research_interests, email, ss_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM professors WHERE semantic_scholar_id = ?", (ss_id,))
    existing = c.fetchone()
    if existing:
        conn.close()
        return existing["id"]
    c.execute("""
        INSERT INTO professors (name, affiliation, department, research_interests, email, semantic_scholar_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, affiliation, department, research_interests, email, ss_id))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def get_professor(prof_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM professors WHERE id = ?", (prof_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def mark_contacted(prof_id):
    conn = get_conn()
    conn.execute("UPDATE professors SET contacted = 1 WHERE id = ?", (prof_id,))
    conn.commit()
    conn.close()


def update_professor_email(prof_id, email):
    conn = get_conn()
    conn.execute("UPDATE professors SET email = ? WHERE id = ?", (email, prof_id))
    conn.commit()
    conn.close()


def list_contacted():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM professors WHERE contacted = 1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_draft(professor_id, subject, body):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO drafts (professor_id, subject, body, status)
        VALUES (?, ?, ?, 'pending')
    """, (professor_id, subject, body))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def list_drafts(status=None):
    conn = get_conn()
    if status:
        rows = conn.execute("""
            SELECT drafts.*, professors.name as professor_name, professors.email as professor_email
            FROM drafts JOIN professors ON drafts.professor_id = professors.id
            WHERE drafts.status = ?
            ORDER BY drafts.created_at DESC
        """, (status,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT drafts.*, professors.name as professor_name, professors.email as professor_email
            FROM drafts JOIN professors ON drafts.professor_id = professors.id
            ORDER BY drafts.created_at DESC
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_draft(draft_id):
    conn = get_conn()
    row = conn.execute("""
        SELECT drafts.*, professors.name as professor_name, professors.email as professor_email
        FROM drafts JOIN professors ON drafts.professor_id = professors.id
        WHERE drafts.id = ?
    """, (draft_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_draft(draft_id, subject, body):
    conn = get_conn()
    conn.execute("UPDATE drafts SET subject=?, body=? WHERE id=?", (subject, body, draft_id))
    conn.commit()
    conn.close()


def mark_draft_sent(draft_id, gmail_draft_id):
    conn = get_conn()
    conn.execute("""
        UPDATE drafts SET status='in_gmail', gmail_draft_id=? WHERE id=?
    """, (gmail_draft_id, draft_id))
    conn.commit()
    conn.close()
