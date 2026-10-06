"""Build (or rebuild) the Chroma index from the PDFs in docs/."""

from django.core.management.base import BaseCommand

from core.services.rag.ingestion import run_ingestion


class Command(BaseCommand):
    help = "Embed docs/*.pdf into the Chroma vector store via Cloudflare Workers AI"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Re-embed even if the index has data")

    def handle(self, *args, **options):
        run_ingestion(force=options["force"])
