import fitz
import difflib
import argparse
import sys
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import spacy

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the HF_TOKEN variable
hf_token = os.getenv("HF_TOKEN")


# Add this block for interactive execution in notebook environments
if "ipykernel" in sys.modules:
    sys.argv = [
        "pdf_compare.py",
        "original_pdf.pdf",
        "modified_pdf.pdf",
        "--output",
        "annotated_output.pdf",
        "--summary",
        "change_summary.txt",
    ]


@dataclass
class TextBlock:
    text: str
    bbox: tuple
    page_num: int
    embedding: np.ndarray = None


class PDFComparer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text using the language model."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks from PDF with position information and generate embeddings."""
        doc = fitz.open(pdf_path)
        blocks = []

        for page in doc:
            page_blocks = page.get_text("dict")["blocks"]
            for block in page_blocks:
                if "lines" in block:
                    text = " ".join(
                        [
                            span["text"]
                            for line in block["lines"]
                            for span in line["spans"]
                        ]
                    )
                    if text.strip():
                        text_block = TextBlock(
                            text=text, bbox=block["bbox"], page_num=page.number
                        )
                        blocks.append(text_block)

        # Generate embeddings for all blocks
        for block in blocks:
            block.embedding = self.generate_embedding(block.text)

        return blocks

    def find_semantic_matches(
        self,
        original_block: TextBlock,
        modified_blocks: List[TextBlock],
        threshold: float = 0.8,
    ) -> List[Tuple[TextBlock, float]]:
        """Find semantically similar blocks using embeddings."""
        similarities = [
            cosine_similarity(original_block.embedding, block.embedding)[0][0]
            for block in modified_blocks
        ]

        matches = [
            (block, sim)
            for block, sim in zip(modified_blocks, similarities)
            if sim > threshold
        ]

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def analyze_semantic_changes(
        self, original_blocks: List[TextBlock], modified_blocks: List[TextBlock]
    ) -> List[Dict]:
        """Analyze changes with semantic understanding."""
        changes = []

        for orig_block in original_blocks:
            matches = self.find_semantic_matches(orig_block, modified_blocks)

            if not matches:
                # Content was deleted
                changes.append(
                    {
                        "type": "Deletion",
                        "original": orig_block.text,
                        "bbox": orig_block.bbox,
                        "page": orig_block.page_num,
                        "confidence": 1.0,
                    }
                )
            else:
                best_match, similarity = matches[0]
                if similarity < 0.95:  # Threshold for considering it a modification
                    # Content was modified
                    changes.append(
                        {
                            "type": "Modification",
                            "original": orig_block.text,
                            "modified": best_match.text,
                            "bbox": best_match.bbox,
                            "page": best_match.page_num,
                            "confidence": similarity,
                        }
                    )

        # Find insertions
        for mod_block in modified_blocks:
            if not any(
                mod_block.text == change.get("modified", "") for change in changes
            ):
                matches = self.find_semantic_matches(mod_block, original_blocks)
                if not matches:
                    changes.append(
                        {
                            "type": "Insertion",
                            "modified": mod_block.text,
                            "bbox": mod_block.bbox,
                            "page": mod_block.page_num,
                            "confidence": 1.0,
                        }
                    )

        return changes

    def highlight_changes_in_pdf(
        self, pdf_path: str, changes: List[Dict], output_path: str
    ):
        """Create annotated PDF with highlighted changes."""
        doc = fitz.open(pdf_path)

        colors = {
            "Deletion": (1, 0.7, 0.7),  # Light red
            "Insertion": (0.7, 1, 0.7),  # Light green
            "Modification": (0.7, 0.7, 1),  # Light blue
        }

        for change in changes:
            if change["type"] != "Deletion":
                page = doc[change["page"]]
                rect = fitz.Rect(change["bbox"])

                # Add highlight
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=colors[change["type"]])
                highlight.update()

                # Add comment
                comment = self._generate_change_comment(change)
                annot = page.add_text_annot(fitz.Point(rect.x1 + 5, rect.y0), comment)
                annot.set_info(title=change["type"])
                annot.update()

        doc.save(output_path)
        doc.close()

    def _generate_change_comment(self, change: Dict) -> str:
        """Generate detailed comment for change annotation."""
        if change["type"] == "Modification":
            return (
                f"Modified (confidence: {change['confidence']:.2f}):\n"
                f"From: {change['original']}\n"
                f"To: {change['modified']}"
            )
        elif change["type"] == "Deletion":
            return f"Deleted: {change['original']}"
        else:  # Insertion
            return f"Inserted: {change['modified']}"

    def generate_change_summary(self, changes: List[Dict], output_path: str):
        """Generate detailed change summary with confidence scores."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("PDF Comparison Summary\n")
            f.write("=====================\n\n")

            for i, change in enumerate(changes, 1):
                f.write(f"Change {i}:\n")
                f.write(f"Type: {change['type']}\n")
                f.write(f"Confidence: {change['confidence']:.2f}\n")
                f.write(f"Location: Page {change['page'] + 1}\n")

                if change["type"] == "Modification":
                    f.write("Original Text:\n")
                    f.write(f"{change['original']}\n")
                    f.write("Modified Text:\n")
                    f.write(f"{change['modified']}\n")
                elif change["type"] == "Deletion":
                    f.write("Deleted Text:\n")
                    f.write(f"{change['original']}\n")
                else:  # Insertion
                    f.write("Inserted Text:\n")
                    f.write(f"{change['modified']}\n")

                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Comparison Tool")
    parser.add_argument("original_pdf", help="Path to the original PDF")
    parser.add_argument("modified_pdf", help="Path to the modified PDF")
    parser.add_argument(
        "--output", default="annotated_output.pdf", help="Output path for annotated PDF"
    )
    parser.add_argument(
        "--summary", default="change_summary.txt", help="Output path for change summary"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the language model to use",
    )
    args = parser.parse_args()

    comparer = PDFComparer(model_name=args.model)

    # Extract and analyze text blocks
    original_blocks = comparer.extract_text_blocks(args.original_pdf)
    modified_blocks = comparer.extract_text_blocks(args.modified_pdf)

    # Analyze changes
    changes = comparer.analyze_semantic_changes(original_blocks, modified_blocks)

    # Generate outputs
    comparer.highlight_changes_in_pdf(args.modified_pdf, changes, args.output)
    comparer.generate_change_summary(changes, args.summary)


if __name__ == "__main__":
    main()
