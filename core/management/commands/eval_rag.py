"""Measure the search on evaluation/questions.json: does it find the passage each
question was written from, and does it stay empty for off-topic messages?"""

import json

from django.core.management.base import BaseCommand

from core.services.rag.ingestion import run_ingestion
from core.services.rag.rag_chat import route, search_query
from core.services.rag.retrieval import BASE_DIR, search


class Command(BaseCommand):
    help = "Evaluate retrieval on evaluation/questions.json"

    def handle(self, *args, **options):
        run_ingestion()
        data = json.loads((BASE_DIR / "evaluation" / "questions.json").read_text(encoding="utf-8"))

        page_hits = pdf_hits = 0
        for item in data["questions"]:
            docs = search(search_query(item["question"], []))
            same_pdf = [doc for doc in docs if doc.metadata["source"] == item["source"]]
            pdf_hits += bool(same_pdf)
            page_hits += any(abs(doc.metadata["page"] - item["page"]) <= 1 for doc in same_pdf)
        total = len(data["questions"])
        self.stdout.write(f"Questions: {total}")
        self.stdout.write(f"  source page found (±1): {page_hits}/{total}")
        self.stdout.write(f"  source PDF found:       {pdf_hits}/{total}")

        answered = [
            message for message in data["off_topic"]
            if route(message, []) == "question" and search(search_query(message, []))
        ]
        self.stdout.write(
            f"Off-topic messages that still get documents: {len(answered)}/{len(data['off_topic'])} {answered}"
        )
