"""Exercise 2 — HITL with interrupt() + Command(resume=...).

Starting from exercise 1, turn the `human_approval` node from a placeholder
into real HITL: call interrupt() with a payload containing diff + reasoning,
then resume the graph with Command(resume=<user choice>).
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": "Senior reviewer. Structured output."},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_human_approval(state: ReviewState) -> dict:
    """Pause and ask the human."""
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {
        "human_choice": response["choice"],
        "human_feedback": response.get("feedback"),
    }


def _render_comment_body(state: ReviewState) -> str:
    """Build the Markdown comment body posted back to the PR."""
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    return "\n".join(lines)


def _post(state: ReviewState, label: str) -> str:
    """Post the review comment to the PR. Returns the final_action string."""
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]→ commit[/cyan]")
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def node_escalate(state):     return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("escalate", node_escalate), ("commit", node_commit),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=MemorySaver())


def prompt_human(payload: dict) -> dict:
    console.print(Panel.fit(
        f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
        f"[dim]{payload['confidence_reasoning']}[/dim]\n\n"
        f"[bold]Summary:[/bold] {payload['summary']}",
        title="Approval request",
        border_style="green",
    ))
    for c in payload.get("comments", []):
        console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
    if payload.get("diff_preview"):
        console.print("\n[dim]--- diff preview ---[/dim]")
        console.print(payload["diff_preview"])

    choice = ""
    while choice not in {"approve", "reject", "edit"}:
        choice = console.input("\n[bold]Choice (approve/reject/edit)?[/bold] ").strip().lower()
    feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
    return {"choice": choice, "feedback": feedback}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 2 — HITL with interrupt()[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = prompt_human(payload)
        result = app.invoke(Command(resume=answer), cfg)

    console.rule("Done")
    console.print(result.get("final_action"))


if __name__ == "__main__":
    main()
