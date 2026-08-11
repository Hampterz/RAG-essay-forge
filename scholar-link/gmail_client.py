"""
Gmail integration — creates real drafts in the user's Gmail account via the
Gmail API. It NEVER sends automatically; drafts land in Gmail's Drafts
folder for the user to review and send themselves, same as ScholarLink.

Setup (one-time, all free):
  1. Go to https://console.cloud.google.com/ and create a project.
  2. Enable the "Gmail API" for that project.
  3. Configure the OAuth consent screen as "External" + "Testing" mode,
     add your own Google account as a test user. Testing mode never
     expires for you and needs no Google review since you're the only user.
  4. Create OAuth credentials -> "Desktop app" -> download the JSON,
     save it as credentials/client_secret.json in this project.
  5. First time you create a draft, a browser window opens for you to
     approve access. A token.json is cached after that so you won't be
     asked again.

Scope used: gmail.compose — this only allows creating/editing drafts,
NOT reading your inbox or sending mail directly, which is the minimum
permission needed and keeps things easy to reason about.
"""

import base64
import os
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
CLIENT_SECRET_FILE = "credentials/client_secret.json"
TOKEN_FILE = "credentials/token.json"


def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET_FILE):
                raise FileNotFoundError(
                    f"Missing {CLIENT_SECRET_FILE}. See gmail_client.py docstring "
                    "for setup steps (Google Cloud Console -> OAuth client -> Desktop app)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def create_gmail_draft(to_email, subject, body):
    service = get_gmail_service()

    message = MIMEText(body)
    message["to"] = to_email
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    draft = service.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw}},
    ).execute()

    return draft["id"]
