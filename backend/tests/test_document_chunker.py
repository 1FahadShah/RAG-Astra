
import unittest

from app.services.document_chunker import chunk_document
from app.services.document_parser import (
    ExtractedUnit,
    ExtractionResult,
)


class DocumentChunkerTests(unittest.TestCase):
    def make_extraction(self, *units):
        return ExtractionResult(
            source_filename="sample.txt",
            source_format="txt",
            units=tuple(units),
        )

    def test_short_unit_creates_one_chunk(self):
        extraction = self.make_extraction(
            ExtractedUnit(text="Short text.", unit_index=0)
        )

        chunks = chunk_document(extraction)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short text.")
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_long_unit_is_split(self):
        text = " ".join(["word"] * 100)
        extraction = self.make_extraction(
            ExtractedUnit(text=text, unit_index=0)
        )

        chunks = chunk_document(
            extraction,
            chunk_size=100,
            overlap=10,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.char_count <= 100 for chunk in chunks))

    def test_preserves_page_and_section_provenance(self):
        extraction = self.make_extraction(
            ExtractedUnit(
                text=" ".join(["evidence"] * 30),
                unit_index=0,
                page_number=7,
                section_title="Security",
            )
        )

        chunks = chunk_document(
            extraction,
            chunk_size=80,
            overlap=10,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_number == 7 for chunk in chunks))
        self.assertTrue(
            all(chunk.section_title == "Security" for chunk in chunks)
        )

    def test_chunk_indices_are_sequential_across_units(self):
        extraction = self.make_extraction(
            ExtractedUnit(text="First unit.", unit_index=0),
            ExtractedUnit(text="Second unit.", unit_index=1),
        )

        chunks = chunk_document(extraction)

        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )

    def test_empty_extraction_returns_no_chunks(self):
        extraction = self.make_extraction()

        self.assertEqual(chunk_document(extraction), [])

    def test_rejects_invalid_chunk_size(self):
        extraction = self.make_extraction(
            ExtractedUnit(text="Some text.", unit_index=0)
        )

        with self.assertRaises(ValueError):
            chunk_document(extraction, chunk_size=0)

    def test_rejects_negative_overlap(self):
        extraction = self.make_extraction(
            ExtractedUnit(text="Some text.", unit_index=0)
        )

        with self.assertRaises(ValueError):
            chunk_document(extraction, overlap=-1)

    def test_rejects_overlap_equal_to_chunk_size(self):
        extraction = self.make_extraction(
            ExtractedUnit(text="Some text.", unit_index=0)
        )

        with self.assertRaises(ValueError):
            chunk_document(extraction, chunk_size=100, overlap=100)

    def test_chunks_never_exceed_configured_size(self):
        text = "abcdefghijklmnopqrstuvwxyz " * 20
        extraction = self.make_extraction(
            ExtractedUnit(text=text, unit_index=0)
        )

        chunks = chunk_document(
            extraction,
            chunk_size=50,
            overlap=10,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(chunk.text) <= 50 for chunk in chunks)
        )

    def test_overlap_preserves_shared_text_between_chunks(self):
        text = " ".join(f"word{i}" for i in range(40))
        extraction = self.make_extraction(
            ExtractedUnit(text=text, unit_index=0)
        )

        chunks = chunk_document(
            extraction,
            chunk_size=50,
            overlap=10,
        )

        self.assertGreater(len(chunks), 1)

        for previous, current in zip(chunks, chunks[1:]):
            self.assertTrue(
                set(previous.text.split()) & set(current.text.split()),
                "Adjacent chunks should share some words.",
            )

if __name__ == "__main__":
    unittest.main()
