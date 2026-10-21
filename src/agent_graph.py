"""
Phase 3.b + 4: The agent graph -- nodes, edges and the decision node.

- Each node is a plain function: (state) -> partial state update. LangGraph merges the update into the running state.
This is the core mental model.

- classify_node and decide_node both call an LLM, but for different jobs. Classification is a light weight, cheap and
structured operation i.e., it depends on (category+urgency). Decision is the expensive operation which is a reasoning
step say what to DO along with the justification. So, when any of these nodes calls the LLM, to which LLM it is being
called is taken care by llm_provider.py but not by the agent_graph.py because this file's implementation is moreover
on how does the nodes and the edges are implemented and not related to the LLM.

- Conditional Edges: guardrail_node is responsible for the agent to move it to the next state and this helps if any 
intervention is needed by the humans for any approvals. So, this node literally decides and changes which node should
run next and make it to perform any action as a next step in the respective node or escalate it to the humans for any
approval or other input. Because of this reason it becomes as an agent but not a pipeline. As you must be aware by
this point that LLM makes the decision of which action should happen next, the control flow of the whole agent is 
purely depends on the LLM's own output.

- Node: Each node is considered as a function. For ex, LLM calling or passing the flow to the next node.
"""

import json

from langgraph.graph import StateGraph, END
from src.agent_state import EmailState
from src.gmail_tools import label_mail, archive_mail, create_draft_mail, send_reply_mail, DraftReplyInput
from src import guardrails, memory
from src.llm_provider import get_llm

# Use the provider abstraction (defaults to Ollama, falls back to Anthropic if configured) [cite:1]
llm = get_llm(temperature = 0)

def classify_node(state: EmailState) -> dict:
    """Cheap structured classification: category + urgency."""
    
    prompt = f"""
        Classify this email. Respond with ONLY a JSON object, no other text.
        Subject: {state['subject']}
        From: {state['sender']}
        Body: {state['body'][:1000]}

        {{
                "category": one of ["customer_inquiry","internal","newsletter","spam","personal","action_required"],
                "urgency": one of ["low","medium","high"]
        }}
    """

    resp = llm.invoke(prompt)
    parsed = json.loads(resp.content.strip().strip("```json").strip("```"))
    return {"category": parsed["category"], "urgency": parsed["urgency"]}


def decide_node(state: EmailState) -> dict:
    """The core reasoning step: what should the agent DO and how confident is it?"""

    history_context = memory.summarize_history_for_prompt(state["sender"]) or "No prior history with this sender."

    prompt = f"""
        Act as an email triage agent. Decide the single best action for this email.
        Subject: {state["subject"]}
        From: {state["sender"]}
        Category: {state["category"]} | Urgency: {state["urgency"]}
        Body: {state["body"][:1500]}

        {history_context}

        Respond with ONLY a JSON object, no other text:
        {{
            "action": one of ["reply","draft","label","archive","escalate","ignore"],
            "reasoning": "one sentence justification for why the action has been chosen",
            "confidence": a number 0.0-1.0,
            "draft_body": "reply text if action is draft/reply, else empty string",
            "label_name": "label to apply if action is label, else empty string"
        }}
    """

    resp = llm.invoke(prompt)
    parsed = json.loads(resp.content.strip().strip("```json").strip("```"))
    return {
        "action": parsed["action"],
        "reasoning": parsed["reasoning"],
        "confidence": float(parsed["confidence"]),
        "draft_body": parsed.get("draft_body", ""),
    }


def guardrail_node(state: EmailState) -> dict:
    """The single checkpoint before any destructive action executes."""
    approved, reason = guardrails.evaluate(
        state["action"], state["confidence"], state["subject"], state["body"]
    )
    return {"approved": approved, "escalation_reason": reason}


def act_node(state: EmailState) -> dict:
    """Executes the approved action via Gmail tools, then log to memory."""    
    action = state["action"]
    result = "no-op"

    if not state["approved"] and action in ("reply", "archive"):
        # Here, not approved (false). Guardrail blocked it - escalate instead of executing.
        result = label_mail.invoke({
            "message_id": state["message_id"], "label_name": "NeedsReview"
        })
    elif action == "reply":
        result = send_reply_mail.invoke({
            "input": DraftReplyInput(
                thread_id = state["thread_id"],
                message_id = state["message_id"],
                to = state["sender"],
                subject = f"Re: {state["subject"]}",
                body = state["body"],
            )
        })
    elif action == "draft":
        result = create_draft_mail.invoke({
            "input": DraftReplyInput(
                thread_id = state["thread_id"],
                message_id = state["message_id"],
                to = state["sender"],
                subject = f"Re: {state["subject"]}",
                body = state["body"],
            )
        })
    elif action == "label":
        result = label_mail.invoke({
            "message_id": state["message_id"],
            "label_name": state["category"]
        })
    elif action == "archive":
        result = archive_mail.invoke({
            "message_id": state["message_id"]
        })
    elif action == "escalate":
        result = label_mail.invoke({
            "message_id": state["message_id"],
            "label_name": "NeedsReview"
        })
    
    memory.log_decision(
        thread_id = state["thread_id"],
        sender = state["sender"],
        subject = state["subject"],
        category = state["category"],
        action = action,
        reasoning = state["reasoning"],
        confidence = state["confidence"],
        result = result,
    )
    return {"result": result}

def route_after_guardrail(state: EmailState) -> str:
    """Conditional edge: this is what makes control flow depene on the LLM's own reasoning + the guardrail check,
    rather than being a fixed pipeline."""
    return "act" # act_node itself branches to escalation internally

def build_graph():
    graph = StateGraph(EmailState)
    graph.add_node("classify", classify_node)
    graph.add_node("decide", decide_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("act", act_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "decide")
    graph.add_edge("decide", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"act": "act"})
    graph.add_edge("act", END)

    return graph.compile()

if __name__ == "__main__":
    # 1. Define a mock email state for testing
    test_state = {
        "message_id": "msg_12345",
        "thread_id": "thread_abcde",
        "sender": "test.user@example.com",
        "subject": "Inquiry about our recent invoice",
        "body": "Hello, I have a question regarding the billing details on our latest invoice. Please review and reply.",
        "category": "",
        "urgency": "",
        "action": "",
        "reasoning": "",
        "confidence": 0.0,
        "draft_body": "",
        "label_name": "",
        "approved": False,
        "escalation_reason": "",
        "result": ""
    }

    print("🚀 Building and compiling agent graph...")
    agent = build_graph()

    print("\n▶️ Invoking agent graph with test email...")
    final_state = agent.invoke(test_state)

    print("\n✅ Execution Complete! Final State:")
    print(json.dumps(final_state, indent=2))