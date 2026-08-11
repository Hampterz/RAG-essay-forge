from flask import Flask, render_template, request, redirect, url_for, flash

import db
import scholar
import llm
import gmail_client

app = Flask(__name__)
app.secret_key = "dev-secret-change-me"  # fine for a local-only app

db.init_db()


@app.route("/")
def index():
    drafts = db.list_drafts(status="pending")
    contacted = db.list_contacted()
    return render_template("index.html", drafts=drafts, contacted_count=len(contacted))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        db.save_profile(
            name=request.form["name"],
            background=request.form["background"],
            interests=request.form["interests"],
            resume_text=request.form["resume_text"],
            email_template=request.form["email_template"],
            gemini_api_key=request.form.get("gemini_api_key", ""),
            gemini_model=request.form.get("gemini_model", ""),
        )
        flash("Profile saved.")
        return redirect(url_for("profile"))

    return render_template("profile.html", profile=db.get_profile())


@app.route("/api/models", methods=["POST"])
def api_models():
    api_key = request.json.get("api_key")
    if not api_key:
        return {"error": "No API key provided"}, 400
    try:
        models = llm.get_gemini_models(api_key)
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        try:
            results = scholar.search_authors(query, limit=10)
        except Exception as e:
            flash(f"Search failed: {e}")
    return render_template("search.html", results=results, query=query)


@app.route("/prepare_draft/<author_id>")
def prepare_draft(author_id):
    """Pull author detail from Semantic Scholar, save as a professor record,
    generate a draft with the local LLM, and drop it in the queue."""
    profile = db.get_profile()
    if not profile["name"]:
        flash("Fill out your profile first.")
        return redirect(url_for("profile"))

    detail = scholar.get_author_detail(author_id)
    affiliation = detail["affiliations"][0] if detail["affiliations"] else "Unknown affiliation"

    prof_id = db.upsert_professor(
        name=detail["name"],
        affiliation=affiliation,
        department="",  # Semantic Scholar doesn't expose department directly
        research_interests=detail["research_summary"],
        email="",  # not provided by Semantic Scholar — see README
        ss_id=author_id,
    )
    professor = db.get_professor(prof_id)

    if professor["contacted"]:
        flash(f"You've already contacted {professor['name']}.")
        return redirect(url_for("search"))

    try:
        subject, body = llm.generate_draft(profile, professor, profile["email_template"])
    except Exception as e:
        flash(f"Draft generation failed: {e}")
        return redirect(url_for("search"))

    db.create_draft(prof_id, subject, body)
    flash(f"Draft prepared for {professor['name']}. Review it in your queue.")
    return redirect(url_for("queue"))


@app.route("/queue")
def queue():
    drafts = db.list_drafts(status="pending")
    return render_template("drafts.html", drafts=drafts)


@app.route("/draft/<int:draft_id>", methods=["GET", "POST"])
def edit_draft(draft_id):
    if request.method == "POST":
        db.update_draft(draft_id, request.form["subject"], request.form["body"])
        if "professor_email" in request.form:
            d = db.get_draft(draft_id)
            db.update_professor_email(d["professor_id"], request.form["professor_email"])
        flash("Draft updated.")
        return redirect(url_for("edit_draft", draft_id=draft_id))

    d = db.get_draft(draft_id)
    return render_template("draft_detail.html", draft=d)

@app.route("/draft/<int:draft_id>/regenerate", methods=["POST"])
def regenerate_draft(draft_id):
    d = db.get_draft(draft_id)
    profile = db.get_profile()
    professor = db.get_professor(d["professor_id"])
    try:
        subject, body = llm.generate_draft(profile, professor, profile["email_template"])
        db.update_draft(draft_id, subject, body)
        flash("Draft regenerated successfully.")
    except Exception as e:
        flash(f"Regeneration failed: {e}")
    return redirect(url_for("edit_draft", draft_id=draft_id))


@app.route("/draft/<int:draft_id>/send_to_gmail")
def send_to_gmail(draft_id):
    """Creates the draft inside the user's actual Gmail Drafts folder.
    Does NOT send it — the user still sends it themselves from Gmail."""
    d = db.get_draft(draft_id)
    to_email = d["professor_email"] or ""
    if not to_email:
        flash(
            "This professor has no email on file (Semantic Scholar doesn't provide one). "
            "Add it manually on the draft page before pushing to Gmail."
        )
        return redirect(url_for("edit_draft", draft_id=draft_id))

    try:
        gmail_draft_id = gmail_client.create_gmail_draft(to_email, d["subject"], d["body"])
    except Exception as e:
        flash(f"Gmail draft creation failed: {e}")
        return redirect(url_for("edit_draft", draft_id=draft_id))

    db.mark_draft_sent(draft_id, gmail_draft_id)
    db.mark_contacted(d["professor_id"])
    flash("Draft created in your Gmail — go review and send it from there.")
    return redirect(url_for("queue"))


@app.route("/contacted")
def contacted():
    professors = db.list_contacted()
    return render_template("contacted.html", professors=professors)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
