"""
Phase 1: Gmail OAuth2 authentication.
 
WHAT IT IS DONE HERE?:
- OAuth2 "installed app" flow (browser consent → local redirect → token).
- Why we request specific SCOPES (principle of least privilege — an agent
  that only needs to read+modify shouldn't request full account access).
- Token persistence + silent refresh, so the agent doesn't re-prompt you
  for consent every single run.
 
This is the ONE piece of this project that cannot be run inside a sandbox —
it needs a real browser and a real Google Cloud OAuth client. Everything
downstream (tools, graph, memory) is provider-agnostic and testable with
the mock inbox in tests/test_agent_mock.py.
"""

# ====================================================================================
# 1. PYTHON SCRIPT -> GMAILAPI CONNECTION
# Firstly, in order for your python gmail_auth script to connect to your GmailAPI, 
# you should have some credentials in place. Check for the credentials file in your
# project folder structure.
# ====================================================================================
import os
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ====================================================================================
# 2. ACCESS PRIVILEGES DEFINITION
# Defining the SCOPES (least privileges) to access the mails from the gmail account.
# So, 'modify' access will take care of reading/modifying the mails.
# ====================================================================================
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / 'token.json'
CREDENTIALS_FILE = PROJECT_ROOT / 'credentials.json'

def get_gmail_service():
    credentials = None
    
    # 3. CREDENTIALS FILE CHECK & INITIATE THE API ACCESS CHECK
    if os.path.exists(TOKEN_FILE):
        print("Found token.json. Loading credentials....")
        credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # If there are no credentials loaded or the loaded credentials are not valid then it'd do either refresh the token or launch browser for log in.
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print('Token expired. Silently refreshing the token using credentials.refresh_token....')
            credentials.refresh(Request())
        else:
            print('No credentials exists. Launching browser for OAuth.')
            # Installed App Flow: This takes care of launching the browser for user consent manually.
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
        
        # Credentials fetched above and saved in token.json (if this file doesn't exists then it creates one)
        print('Saving the credentials to token.json')
        with TOKEN_FILE.open("w") as token:
            token.write(credentials.to_json())
    return build('gmail', 'v1', credentials=credentials)

if __name__ == "__main__":
    service = get_gmail_service()
    profile = service.users().getProfile(userId='me').execute()
    print("\nSUCCESS! Authenticated successfully.")
    print(f"Connected Gmail Account: {profile.get('emailAddress')}")
    print(f"Total Messages in Inbox: {profile.get('messagesTotal')}")