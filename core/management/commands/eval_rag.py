"""Measure the chatbot on evaluation/questions.json: does the search find the passage
each question was written from, does it stay empty for off-topic messages, and are
acknowledgements answered without a search?"""

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

        def documents_for(message, history=()):
            query = search_query(message, list(history))
            return search(query) if query else []

        page_hits = pdf_hits = 0
        for item in data["questions"]:
            docs = documents_for(item["question"])
            same_pdf = [doc for doc in docs if doc.metadata["source"] == item["source"]]
            pdf_hits += bool(same_pdf)
            page_hits += any(abs(doc.metadata["page"] - item["page"]) <= 1 for doc in same_pdf)
        total = len(data["questions"])
        self.stdout.write(f"Questions: {total}")
        self.stdout.write(f"  source page found (±1): {page_hits}/{total}")
        self.stdout.write(f"  source PDF found:       {pdf_hits}/{total}")

        answered = [
            message for message in data["off_topic"]
            if route(message, []) == "question" and documents_for(message)
        ]
        self.stdout.write(
            f"Off-topic messages that still get documents: {len(answered)}/{len(data['off_topic'])} {answered}"
        )

        # Routing after an answer, the situation in which people acknowledge.
        history = [["Qu'est-ce que le droit à un environnement sain ?",
                    "C'est le droit de vivre dans un environnement qui ne nuit pas à la santé."]]
        searched = [m for m in data["acknowledgements"] if route(m, history) != "small_talk"]
        self.stdout.write(
            f"Acknowledgements answered without a search: "
            f"{len(data['acknowledgements']) - len(searched)}/{len(data['acknowledgements'])} {searched}"
        )
        lost = [m for m in data["keep_as_questions"] if route(m, history) != "question"]
        self.stdout.write(
            f"Questions still searched: "
            f"{len(data['keep_as_questions']) - len(lost)}/{len(data['keep_as_questions'])} {lost}"
        )
