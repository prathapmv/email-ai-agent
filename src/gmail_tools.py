"""
Phase 2: Gmail actions exposed as LangChain tools.
 
WHAT IT IS DONE HERE:
- Tool design: each function has a clear name, docstring, and typed schema —
  this docstring IS the prompt the LLM sees when deciding whether to call it.
  Bad docstring = bad tool selection, no matter how good your model is.
- Structured input/output (Pydantic) instead of raw strings, so the agent's
  tool calls are validated before they hit a real API.
- Separating "read" tools (safe, always allowed) from "write" tools (gated
  by guardrails.py before execution).
"""

import base64
from typing import List
from pydantic import BaseModel, Field
from langchain.tools import tool
from src.gmail_auth import get_gmail_service
from email.mime.text import MIMEText

_service = None

def _svc():
    global _service
    if _service is None:
        _service = get_gmail_service()
    return _service

# ====================================================================================
# 1. SCHEMA DEFINITION
# We're going to use Pydantic (a python library) to structure the data fetched from
# Gmail app basically the mails. This pydantic structured/modeled data is ready to 
# send it to LLM to process by the LLM model for reasoning and deciding which 
# action(s) to take. Later, the output provided by the LLMs is VALIDATED against 
# this schema.
# ====================================================================================

class EmailSummary(BaseModel):
    thread_id: str
    message_id: str
    sender: str
    subject: str
    snippet: str
    body: str

class DraftReplyInput(BaseModel):
    thread_id: str = Field(description="Gmail thread ID to reply within")
    message_id: str = Field(description="Gmail message ID of the soecific email being replied to")
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Subject line, usually 'Re: original subject'")
    body: str = Field(description="Plain-text reply body")


# 2. EMAIL BODY EXTRACTION
# Mostly, for each mail processing, we need to scan through the actual content of the 
# mail i.e., the subject. This function is responsible for extracting the subject of
# each mail.

def _extract_body(payload) -> str:
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        # 'data' -> represents that certain layer in the email contains body with
        # the key as 'data'. ('data' in) check prevents the script from error when
        # it hits an email layer where the body does not contain 'data' field.
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""

# READ TOOLS
# 3. LISTING UNREAD EMAILS
@tool
def list_unread_mails(max_results: int = 10) -> List[EmailSummary]:
    """Fetch recent unread emails from the user's Gmail inbox."""
    svc = _svc()
    unread_mails = svc.users().messages().list(
        userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results
    ).execute()
    unread_mails_for_LLM=[]
    for m in unread_mails.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        unread_mails_for_LLM.append(EmailSummary(
            thread_id=msg["threadId"],
            body=_extract_body(msg["payload"]),
            message_id=msg["id"],
            sender=headers.get("From", ""),
            snippet=msg.get("snippet", ""),
            subject=headers.get("Subject", "")
        ))
    print(unread_mails_for_LLM)
    return unread_mails_for_LLM

# 4. SEARCH EMAILS
@tool
def search_emails(query: str, max_results: int = 5) -> List[EmailSummary]:
    """Search emails using Gmail search syntax (eg: 'from:prathapmv07@gmail.com is:unread').
    Useful for gathering contexr on a sender's history before deciding how to respond."""
    svc = _svc()
    
    resp = svc.users().messages().list(
        userId = "me",
        q = query,
        maxResults = max_results
    ).execute()
    
    searched_mails = []
    for mail in resp.get("messages", []):
        msg = svc.users().messages().get(
            userId = "me",
            id = mail["id"],
            format = "full"
        ).execute()
        headers = {h["name"] : h["value"] for h in msg["payload"]["headers"]}
        searched_mails.append(EmailSummary(
            thread_id = msg["threadId"],
            message_id = msg["id"],
            sender = headers.get("From", ""),
            subject = headers.get("Subject", ""),
            snippet = msg.get("snippet", ""),
            body = _extract_body(msg["payload"])
        ))
    return searched_mails


# WRITE TOOLS
# 5. LABELLING EMAIL
@tool
def label_mail(message_id: str, label_name: str) -> str:
    """Apply a Gmail label to a message (creating the label if it doesn't exist)."""
    svc = _svc()
    labels = svc.users().labels().list(userId = "me").execute().get("labels", [])
    label_id = next((l["id"] for l in labels if l["name"].lower() == label_name.lower()), None)

    if not label_id:
        created = svc.users().labels().create(
            userId = "me", 
            body = {
                "name": label_name
            }
        ).execute()
        label_id = created["id"]
    
    svc.users().messages().modify(
        userId="me", id=message_id, body={"addLabelIds": [label_id]}
    ).execute()
    return f"Labeled {message_id} with '{label_name}'"


# 6. ARCHIVING EMAIL
@tool
def archive_mail(message_id: str) -> str:
    """Archive a message (remove from inbox). DESTRUCTIVE - gated by guardrails."""
    svc = _svc()
    svc.users().messages().modify(
        userId = "me",
        id = message_id,
        body = {
            "removeLabelIds": ["INBOX"]
        }
    ).execute()
    return f"Archived {message_id}"


# 7. SENDING A REPLY EMAIL
@tool
def send_reply_mail(input_reply_mail: DraftReplyInput) -> str:
    """Send a reply immediately. DESTRUCTIVE - only call this after guardrails
    approval (confidence above threshold, not flagged as sensitive)"""
    svc = _svc()
    
    # In order to send a reply mail to a particular mail thread, we need to fetch the original
    # message's real 'Message-ID' header. This is different from the Gmail's threadId.
    source_mail = svc.users().messages().get(
        userId = "me",
        id = input_reply_mail.message_id,
        format = "metadata",
        metadataHeaders = ["Message-ID", "Subject", "References"]
    ).execute()

    headers = {h["name"] : h["value"] for h in source_mail["payload"].get("headers", [])}
    original_msg_id = headers.get("Message-ID","")
    original_subject = headers.get("Subject", "")
    existing_references = headers.get("References", "")

    clean_subject = original_subject.strip()
    if not clean_subject.lower().startswith("re:"):
        original_reply_subject = f"Re: {clean_subject}"
    else:
        original_reply_subject = clean_subject

    # Set the original message ID in the 'mail_to_send' data so that while the mail is sent
    # it'd send to that particular mail thread but not separately.
    mail_to_send = MIMEText(input_reply_mail.body)
    mail_to_send["To"] = input_reply_mail.to
    mail_to_send["Subject"] = original_reply_subject
    
    if original_msg_id:
        mail_to_send["In-Reply-To"] = original_msg_id
        if existing_references:
            mail_to_send["References"] = f"{existing_references} {original_msg_id}"
        else:
            mail_to_send["References"] = original_msg_id

    raw_content = base64.urlsafe_b64encode(mail_to_send.as_bytes()).decode()
    
    sent_mail = svc.users().messages().send(
        userId = "me",
        body = {
            "raw": raw_content,
            "threadId": input_reply_mail.thread_id
        }
    ).execute()
    return f"Sent an email: {sent_mail["id"]}"

# 8. CREATING A DRAFT REPLY MAIL
@tool
def create_draft_mail(input_draft_mail: DraftReplyInput) -> str:
    """Create a DRAFT reply (does not send). Always safe - drafts require human send action in Gmail."""
    svc = _svc()
    mail_to_draft = MIMEText(input_draft_mail.body)
    mail_to_draft["To"] = input_draft_mail.to
    mail_to_draft["Subject"] = input_draft_mail.subject
    raw_content = base64.urlsafe_b64encode(mail_to_draft.as_bytes()).decode()
    draft_mail = svc.users().drafts().create(
        userId = "me",
        body = {
            "message" : {
                "raw": raw_content,
                "threadId": input_draft_mail.thread_id
            }
        }
    ).execute()
    return f"Draft mail created: {draft_mail["id"]}"


READ_TOOLS = [list_unread_mails, search_emails]
WRITE_TOOLS = [label_mail, archive_mail, send_reply_mail, create_draft_mail]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS


#if __name__ == "__main__":
    # unread_emails = list_unread_mails.invoke({"max_results": 5})
    # print(unread_emails)
    
    # labels = label_email.invoke({
    #     "message_id": "1a023ee64f29dc73", 
    #     "label_name": "Email AI Agent"
    # })
    
    # archive_email.invoke({
    #     "message_id": "1a023ee64f29dc73"
    # })

    # sending_input_reply_mail = send_reply.invoke({
    #     "input_reply_mail": DraftReplyInput(
    #         thread_id = "1a0040b26c92349c",
    #         message_id = "1a0040b26c92349c",
    #         to = "prathapmv07@gmail.com",
    #         subject = "Re: Payment Reminder",
    #         body = "Just testing the sending reply mail tool"
    #     )
    # })
    # print(sending_input_reply_mail)

    # creating_draft_mail = create_draft_reply.invoke({
    #     "input_draft_mail": DraftReplyInput(
    #         thread_id = "1a0040b26c92349c",
    #         message_id = "1a0040b26c92349c",
    #         to = "prathapmv07@gmail.com",
    #         subject = "Re: Payment Reminder",
    #         body = "Drafted a mail successfully!"
    #     )
    # })
    # print(create_draft_reply)

    # search_results = search_emails.invoke({
    # "query": "from:no-reply@amazonpay.in is:unread",
    # "max_results": 3
    # })
    # print(search_results)