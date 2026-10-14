
import hashlib
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.main import app
from app.models.document import Document
from app.services.document_storage import delete_stored_document


def build_test_pdf(marker: str) -> bytes:
    """Build a small PDF with a unique marker and valid cross-reference."""

    content = (
        f"BT /F1 12 Tf 20 100 Td "
        f"(RAG-Astra integration check {marker}) Tj ET"
    ).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 300 144] "
            b"/Resources << /Font << /F1 5 0 R >> >> "
            b"/Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    pdf.extend(f"% Integration marker {marker}\n".encode("ascii"))
    offsets = [0]

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    return bytes(pdf)


def main() -> None:
    marker = uuid.uuid4().hex
    payload = build_test_pdf(marker)
    checksum = hashlib.sha256(payload).hexdigest()

    settings = get_settings()
    storage_root = settings.storage_dir.resolve()

    uploaded_id = None
    uploaded_storage_key = None

    print("Starting live RAG-Astra upload integration check.")

    try:
        with TestClient(app) as client:
            # 1. Upload through the real API and database dependency.
            response = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        f"integration-{marker}.pdf",
                        payload,
                        "application/pdf",
                    )
                },
            )

            print(f"Upload HTTP status: {response.status_code}")
            assert response.status_code == 201, response.text

            uploaded_id = uuid.UUID(response.json()["id"])
            assert response.json()["checksum_sha256"] == checksum

            # 2. Verify the committed row and file using a real DB session.
            factory = get_session_factory()

            with factory() as db:
                record = (
                    db.query(Document)
                    .filter(Document.id == uploaded_id)
                    .one()
                )

                assert record.checksum_sha256 == checksum
                assert record.storage_key
                assert record.status == "pending"

                uploaded_storage_key = record.storage_key
                stored_path = storage_root / uploaded_storage_key

                assert stored_path.is_file()
                assert stored_path.read_bytes() == payload

            print("Database metadata and stored file: verified.")

            # 3. Verify the list endpoint exposes the new document.
            listing = client.get(
                "/api/v1/documents",
                params={"limit": 100, "offset": 0},
            )

            assert listing.status_code == 200, listing.text

            listed_ids = {
                item["id"] for item in listing.json()["items"]
            }

            assert str(uploaded_id) in listed_ids, (
                "The uploaded document was not found in the first "
                "100 items returned by the listing endpoint."
            )

            print("Document listing endpoint: verified.")

            # 4. The identical upload must be rejected.
            files_before_duplicate = (
                {p.name for p in storage_root.iterdir()}
                if storage_root.exists()
                else set()
            )

            duplicate = client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        f"integration-duplicate-{marker}.pdf",
                        payload,
                        "application/pdf",
                    )
                },
            )

            print(f"Duplicate upload HTTP status: {duplicate.status_code}")
            assert duplicate.status_code == 409, duplicate.text

            files_after_duplicate = (
                {p.name for p in storage_root.iterdir()}
                if storage_root.exists()
                else set()
            )

            assert files_after_duplicate == files_before_duplicate, (
                "Duplicate upload left an extra stored file."
            )

            print("Duplicate rejection and file cleanup: verified.")
            print("LIVE INTEGRATION CHECK PASSED.")

    finally:
        # Delete only the record created by this test run.
        if uploaded_id is not None:
            factory = get_session_factory()

            with factory() as db:
                record = db.get(Document, uploaded_id)

                if record is not None:
                    uploaded_storage_key = record.storage_key
                    db.delete(record)
                    db.commit()

            if uploaded_storage_key:
                delete_stored_document(uploaded_storage_key)

            print("Integration test record and file cleanup completed.")


if __name__ == "__main__":
    main()
