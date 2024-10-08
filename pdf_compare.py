from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, Pipeline, pipeline
import fitz
import spacy
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TextBlock:
    text: str
    bbox: tuple
    page_num: int
    embedding: np.ndarray = None
    context: str = ""
    semantic_hash: str = ""

    def __post_init__(self):
        self.normalized_text = self._normalize_text()

    def _normalize_text(self) -> str:
        text = re.sub(r"\s+", " ", self.text)
        text = text.lower().strip()
        return text

    def __hash__(self):
        return hash((self.normalized_text, self.page_num))


class EnhancedPDFComparer:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        verification_model_name: str = "facebook/bart-large-mnli",
        similarity_threshold: float = 0.95,
        context_window: int = 3,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.verifier = pipeline(
            "zero-shot-classification",
            model=verification_model_name,
            device=self.device,
        )

        self.nlp = spacy.load("en_core_web_sm")

        self.similarity_threshold = similarity_threshold
        self.context_window = context_window

    def extract_text_blocks_with_context(self, pdf_path: str) -> List[TextBlock]:
        doc = fitz.open(pdf_path)
        blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = page.get_text("dict")["blocks"]

            page_text_blocks = []
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
                            text=text, bbox=block["bbox"], page_num=page_num
                        )
                        page_text_blocks.append(text_block)

            for i, block in enumerate(page_text_blocks):
                context_start = max(0, i - self.context_window)
                context_end = min(len(page_text_blocks), i + self.context_window + 1)

                context_blocks = (
                    page_text_blocks[context_start:i]
                    + page_text_blocks[i + 1 : context_end]
                )
                context = " ... ".join([b.text for b in context_blocks])
                block.context = context

                block.semantic_hash = self._generate_semantic_hash(block)
                block.embedding = self._generate_embedding(block)

                blocks.append(block)

        return blocks

    def _generate_semantic_hash(self, block: TextBlock) -> str:
        doc = self.nlp(block.text)

        entities = [ent.text for ent in doc.ents]
        main_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        main_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        fingerprint = f"ENT:{sorted(entities)}|NOUN:{sorted(main_nouns)}|VERB:{sorted(main_verbs)}"
        return fingerprint

    def _generate_embedding(self, block: TextBlock) -> np.ndarray:
        combined_text = f"{block.text} [SEP] {block.context}"
        inputs = self.tokenizer(
            combined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()

    def verify_change(
        self,
        original_block: TextBlock,
        modified_block: Optional[TextBlock],
        change_type: str,
    ) -> Tuple[bool, float]:
        if change_type == "Deletion":
            hypothesis = f"This content was removed: {original_block.text}"
            context = f"Original context: {original_block.context}"
        elif change_type == "Insertion":
            hypothesis = f"This is newly added content: {modified_block.text}"
            context = f"Surrounding context: {modified_block.context}"
        else:  # Modification
            hypothesis = f"This content was modified from '{original_block.text}' to '{modified_block.text}'"
            context = f"Original context: {original_block.context} | New context: {modified_block.context}"

        result = self.verifier(
            context,
            candidate_labels=["true change", "false positive"],
            hypothesis=hypothesis,
        )

        is_verified = result["labels"][0] == "true change"
        confidence = result["scores"][0]

        return is_verified, confidence

    def analyze_changes(
        self, original_blocks: List[TextBlock], modified_blocks: List[TextBlock]
    ) -> List[Dict]:
        changes = []
        used_modified_blocks = set()

        verification_stats = {
            "total_candidates": 0,
            "verified_changes": 0,
            "rejected_candidates": 0,
        }

        for orig_block in original_blocks:
            verification_stats["total_candidates"] += 1

            exact_match = next(
                (
                    mod_block
                    for mod_block in modified_blocks
                    if orig_block.normalized_text == mod_block.normalized_text
                    and mod_block not in used_modified_blocks
                ),
                None,
            )

            if exact_match:
                used_modified_blocks.add(exact_match)
                continue

            semantic_matches = []
            for mod_block in modified_blocks:
                if mod_block in used_modified_blocks:
                    continue

                similarity = float(np.dot(orig_block.embedding, mod_block.embedding.T))
                hash_match = orig_block.semantic_hash == mod_block.semantic_hash

                if similarity > self.similarity_threshold or hash_match:
                    semantic_matches.append((mod_block, similarity, hash_match))

            if semantic_matches:
                semantic_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
                best_match, similarity, hash_match = semantic_matches[0]

                is_verified, confidence = self.verify_change(
                    orig_block, best_match, "Modification"
                )

                if is_verified and confidence > 0.9:
                    verification_stats["verified_changes"] += 1
                    changes.append(
                        {
                            "type": "Modification",
                            "original": orig_block.text,
                            "modified": best_match.text,
                            "bbox": best_match.bbox,
                            "page": best_match.page_num,
                            "confidence": confidence,
                            "verification_method": "semantic_match_verified",
                        }
                    )
                    used_modified_blocks.add(best_match)
                    continue

            is_verified, confidence = self.verify_change(orig_block, None, "Deletion")
            if is_verified and confidence > 0.95:
                verification_stats["verified_changes"] += 1
                changes.append(
                    {
                        "type": "Deletion",
                        "original": orig_block.text,
                        "modified": "",
                        "bbox": orig_block.bbox,
                        "page": orig_block.page_num,
                        "confidence": confidence,
                        "verification_method": "verified_deletion",
                    }
                )
            else:
                verification_stats["rejected_candidates"] += 1

        for mod_block in modified_blocks:
            if mod_block not in used_modified_blocks:
                verification_stats["total_candidates"] += 1
                is_verified, confidence = self.verify_change(
                    None, mod_block, "Insertion"
                )

                if is_verified and confidence > 0.95:
                    verification_stats["verified_changes"] += 1
                    changes.append(
                        {
                            "type": "Insertion",
                            "original": "",
                            "modified": mod_block.text,
                            "bbox": mod_block.bbox,
                            "page": mod_block.page_num,
                            "confidence": confidence,
                            "verification_method": "verified_insertion",
                        }
                    )
                else:
                    verification_stats["rejected_candidates"] += 1

        if changes:
            changes[0]["verification_stats"] = verification_stats

        return changes

    def highlight_changes_in_pdf(
        self, pdf_path: str, changes: List[Dict], output_path: str
    ):
        doc = fitz.open(pdf_path)

        colors = {
            "Deletion": (1, 0.7, 0.7),  # Light red
            "Insertion": (0.7, 1, 0.7),  # Light green
            "Modification": (0.7, 0.7, 1),  # Light blue
        }

        seen_rects = []  # Track which bounding boxes we've already annotated

        for change in changes:
            page = doc[change["page"]]
            rect = fitz.Rect(change["bbox"])

            # Avoid marking overlapping bounding boxes
            if any(self.bbox_overlap(rect, seen_rect) for seen_rect in seen_rects):
                continue

            seen_rects.append(rect)

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

    def bbox_overlap(self, bbox1, bbox2) -> bool:
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        if x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1:
            return False  # No overlap

        return True  # Overlapping

    def _generate_change_comment(self, change: Dict) -> str:
        if change["type"] == "Modification":
            return (
                f"Modified (confidence: {change['confidence']:.2f}):\n"
                f"From: {change['original']}\n"
                f"To: {change['modified']}"
            )
        elif change["type"] == "Deletion":
            return f"Deleted (confidence: {change['confidence']:.2f}): {change['original']}"
        else:  # Insertion
            return f"Inserted (confidence: {change['confidence']:.2f}): {change['modified']}"

    def generate_change_summary(self, changes: List[Dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("PDF Comparison Summary\n")
            f.write("=====================\n\n")

            change_counts = Counter(change["type"] for change in changes)
            f.write("Overall Changes:\n")
            f.write(f"Total changes: {len(changes)}\n")
            f.write(f"Modifications: {change_counts['Modification']}\n")
            f.write(f"Deletions: {change_counts['Deletion']}\n")
            f.write(f"Insertions: {change_counts['Insertion']}\n\n")

            if "verification_stats" in changes[0]:
                stats = changes[0]["verification_stats"]
                f.write("Verification Statistics:\n")
                f.write(f"Total candidates: {stats['total_candidates']}\n")
                f.write(f"Verified changes: {stats['verified_changes']}\n")
                f.write(f"Rejected candidates: {stats['rejected_candidates']}\n\n")

            f.write("Detailed Changes:\n")
            for i, change in enumerate(changes, 1):
                f.write(f"Change {i}:\n")
                f.write(f"Type: {change['type']}\n")
                f.write(f"Confidence: {change['confidence']:.2f}\n")
                f.write(f"Location: Page {change['page'] + 1}\n")
                f.write(
                    f"Verification Method: {change.get('verification_method', 'N/A')}\n"
                )

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

    def compare_pdfs(
        self, original_pdf: str, modified_pdf: str, output_pdf: str, summary_path: str
    ):
        original_blocks = self.extract_text_blocks_with_context(original_pdf)
        modified_blocks = self.extract_text_blocks_with_context(modified_pdf)

        changes = self.analyze_changes(original_blocks, modified_blocks)

        self.highlight_changes_in_pdf(modified_pdf, changes, output_pdf)
        self.generate_change_summary(changes, summary_path)

        return changes


# comparer = EnhancedPDFComparer()
# changes = comparer.compare_pdfs("original_pdf.pdf", "modified_pdf.pdf", "annotated_pdf.pdf", "change_summary.txt")
