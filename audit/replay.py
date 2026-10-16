"""Replay a review session from the audit trail.

Usage:
    uv run python -m audit.replay --thread <thread_id>
    uv run python -m audit.replay --list                # list recent threads

The script reads `audit_events` (human-readable timeline). The LangGraph
SqliteSaver checkpoint tables live in the same .db file but are queried
separately by the LangGraph runtime, not by this tool.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from common.db import db_conn, replay_events


RISK_COLOR = {"low": "green", "med": "yellow", "high": "red"}


async def list_threads() -> None:
    console = Console()
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low'
                   END                   AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()

    table = Table(title="Recent review sessions")
    for col in ("thread_id", "pr_url", "started", "last_event", "worst_risk", "events"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["thread_id"],
            r["pr_url"],
            str(r["started"]),
            str(r["last_event"]),
            str(r["worst_risk"]),
            str(r["events"]),
        )
    console.print(table)


async def replay(thread_id: str) -> None:
    console = Console()
    events = await replay_events(thread_id)

    if not events:
        console.print(f"[red]No events found for thread {thread_id}[/red]")
        return

    console.rule(f"[bold]Replay {thread_id}")
    for ev in events:
        risk = ev["risk_level"]
        risk_colored = f"[{RISK_COLOR.get(risk, 'white')}]{risk:<4}[/]"
        reviewer = ev["reviewer_id"] or "-"
        reason = (ev["reason"] or "")[:60]
        console.print(
            f"[dim]{ev['timestamp']}[/dim]  "
            f"[cyan]{ev['action']:<18}[/cyan] "
            f"conf=[bold]{ev['confidence']:.2f}[/bold] "
            f"risk={risk_colored} "
            f"decision=[magenta]{ev['decision']:<10}[/magenta] "
            f"reviewer={reviewer:<14} "
            f"{ev['execution_time_ms']:>5}ms  "
            f"[dim]{reason}[/dim]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="Replay one thread by id")
    parser.add_argument("--list", action="store_true", help="List recent threads")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_threads())
    elif args.thread:
        asyncio.run(replay(args.thread))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
