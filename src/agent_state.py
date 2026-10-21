"""
Phase 3.a: Agent State Schema (I'd like to see my agent behave in a state manner say on
each state it'll perform certain tasks so that it'll be more organised and easy to track/
debug the execution)

In case of a single prompt mechanism, the tracking and debugging would be challenging.
Hence, I'll be using LangGraph concept which has EXPLICIT states that flows between the
nodes. Each node will has certain read and write fields (receiving the output from a
different node and write an output to it's node for the other node to pick it as an
output for that node respectively). Also, LangGraph handles the memory, time and the 
control flow. By using an explicit state schema and conditional edges, it turns an 
unpredictable LLM into a reliable, auditable state machine that can pause for human review, 
handle errors gracefully, and run autonomously without getting trapped in infinite prompt loops.

This gives us an advantage that you can debug the execution on each node. Say, printing
the output on each node to see what the agent has decided and why (the reason) it has
decided instead of parsing through the chat history (if it's single prompt mechanism).
"""

from typing import Literal, Optional, TypedDict

class EmailState(TypedDict):
    # Fetch Node - defines the state/structure (input received from the read mail tool)
    message_id: str
    thread_id: str
    sender: str
    subject: str
    body: str

    # Classify Node - categorizes the mail. For ex, the categories like 'customer_inquiry', 'spam', 'newsletter' and 
    # the urgency type like 'low', 'medium' and 'high'.
    category: Optional[str]
    urgency: Optional[str]

    # Decide Node - this node decides which action to take. For ex, 'send_mail', 'draft_mail', 'delete_mail', etc.,
    action: Optional[Literal["reply", "draft", "label", "archive", "escalate", "ignore"]]
    reasoning: Optional[str]                # Ollama's stated rationale - critical for auditing an autonomous agent
    confidence: Optional[float]             # 0-1, drives the guardrail decision
    draft_body: Optional[str]               # Will be auto populated with an incoming data if related to reply mail

    # Guardrail Node - this node does the safety check and notifies the user for any approval needed.
    approved: Optional[bool]                # Did the action pass the safety check?
    escalation_reason: Optional[str]

    # Action Node - this node says which action to take
    result: Optional[str]                   # Human readable outcome and this will be logged to memory
