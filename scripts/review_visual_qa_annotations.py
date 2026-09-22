#!/usr/bin/env python3
"""Gradio app for manually accepting or rejecting visual-QA images."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually review visual-QA image annotations")
    parser.add_argument("--annotations", required=True, type=Path, help="Validator JSONL output")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Manual decisions JSONL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            data_id = str(row.get("data_id", ""))
            if not data_id:
                continue
            if data_id not in latest:
                order.append(data_id)
            latest[data_id] = row
    return [latest[data_id] for data_id in order if latest[data_id].get("processing_status") == "success"]


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.decisions: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for row in read_all_jsonl(self.path):
                if row.get("data_id"):
                    self.decisions[str(row["data_id"])] = row

    def save(self, record: dict[str, Any], decision: str, note: str) -> None:
        data_id = str(record["data_id"])
        self.decisions[data_id] = {
            "data_id": data_id,
            "input_query": record.get("input_query"),
            "image_path": record.get("image_path"),
            "image_filename": record.get("image_filename"),
            "manual_decision": decision,
            "manual_note": note.strip(),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm_suitable_for_visual_qa": bool(
                record.get("llm_annotation", {}).get("suitable_for_visual_qa")
            ),
        }
        self._write_atomic()

    def clear(self, data_id: str) -> None:
        self.decisions.pop(data_id, None)
        self._write_atomic()

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in self.decisions.values():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def read_all_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_app(records: list[dict[str, Any]], images_dir: Path, store: ReviewStore) -> gr.Blocks:
    if not records:
        raise ValueError("No successfully processed annotation records to review")

    def clamp(index: int) -> int:
        return max(0, min(int(index), len(records) - 1))

    def stats_markdown() -> str:
        accepted = sum(row.get("manual_decision") == "accept" for row in store.decisions.values())
        rejected = sum(row.get("manual_decision") == "reject" for row in store.decisions.values())
        reviewed = accepted + rejected
        pending = max(0, len(records) - reviewed)
        completion = 100.0 * reviewed / len(records) if records else 0.0
        comparable = [
            row for row in store.decisions.values() if row.get("manual_decision") in {"accept", "reject"}
        ]
        agreements = sum(
            (row["manual_decision"] == "accept") == bool(row.get("llm_suitable_for_visual_qa"))
            for row in comparable
        )
        agreement = 100.0 * agreements / len(comparable) if comparable else 0.0
        return (
            f"### Manual annotation statistics\n"
            f"**Accepted:** {accepted} &nbsp; **Rejected:** {rejected} &nbsp; "
            f"**Pending:** {pending} &nbsp; **Completed:** {reviewed}/{len(records)} ({completion:.1f}%)  \n"
            f"**LLM/manual agreement:** {agreements}/{len(comparable)} ({agreement:.1f}%)"
        )

    def render(index: int):
        index = clamp(index)
        record = records[index]
        data_id = str(record["data_id"])
        filename = record.get("image_filename") or Path(str(record.get("image_path", ""))).name
        image_path = images_dir / filename
        decision = store.decisions.get(data_id, {})
        annotation = record.get("llm_annotation", {})
        header = (
            f"### Item {index + 1} of {len(records)}\n"
            f"**Data ID:** `{data_id}`  \n"
            f"**Query:** {record.get('input_query', '')}  \n"
            f"**Location:** {record.get('location', '')} &nbsp; **Topic:** {record.get('category', '')}  \n"
            f"**LLM decision:** `{annotation.get('status', 'unknown')}` &nbsp; "
            f"**Query relevance:** `{annotation.get('query_relevance', 'unknown')}` &nbsp; "
            f"**Visually grounded:** `{annotation.get('visually_grounded', 'unknown')}` &nbsp; "
            f"**Location relevance:** `{annotation.get('location_relevance', 'unknown')}`"
        )
        return (
            index,
            str(image_path) if image_path.is_file() else None,
            header,
            annotation,
            decision.get("manual_decision", "pending"),
            decision.get("manual_note", ""),
            stats_markdown(),
            str(store.path),
        )

    def navigate(index: int, delta: int):
        return render(clamp(index + delta))

    def record_decision(index: int, decision: str, note: str):
        index = clamp(index)
        store.save(records[index], decision, note)
        next_index = min(index + 1, len(records) - 1)
        return render(next_index)

    def clear_decision(index: int):
        index = clamp(index)
        store.clear(str(records[index]["data_id"]))
        return render(index)

    with gr.Blocks(title="Visual QA Image Review") as demo:
        gr.Markdown("# Visual QA Image Review\nAccept or reject each image after comparing it with the query and LLM assessment.")
        stats = gr.Markdown(stats_markdown())
        index_state = gr.State(0)
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Crawled image", type="filepath", height=560, interactive=False)
            with gr.Column(scale=2):
                item_header = gr.Markdown()
                llm_json = gr.JSON(label="LLM annotation", open=True)
                current_decision = gr.Radio(
                    choices=["pending", "accept", "reject"],
                    value="pending",
                    label="Current manual decision",
                    interactive=False,
                )
                note = gr.Textbox(label="Reviewer note (optional)", lines=3)
        with gr.Row():
            previous = gr.Button("← Previous")
            accept = gr.Button("Accept & Next", variant="primary")
            reject = gr.Button("Reject & Next", variant="stop")
            clear = gr.Button("Clear decision")
            next_button = gr.Button("Next →")
        output_path = gr.Textbox(label="Saved manual annotation file", interactive=False)

        outputs = [index_state, image, item_header, llm_json, current_decision, note, stats, output_path]
        demo.load(render, inputs=index_state, outputs=outputs)
        previous.click(lambda i: navigate(i, -1), inputs=index_state, outputs=outputs)
        next_button.click(lambda i: navigate(i, 1), inputs=index_state, outputs=outputs)
        accept.click(
            lambda i, n: record_decision(i, "accept", n),
            inputs=[index_state, note],
            outputs=outputs,
        )
        reject.click(
            lambda i, n: record_decision(i, "reject", n),
            inputs=[index_state, note],
            outputs=outputs,
        )
        clear.click(clear_decision, inputs=index_state, outputs=outputs)
    return demo


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.annotations.resolve())
    images_dir = args.images_dir.resolve()
    store = ReviewStore(args.output)
    demo = build_app(records, images_dir, store)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(images_dir), str(store.path.parent)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
