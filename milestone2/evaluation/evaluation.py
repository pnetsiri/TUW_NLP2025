"""
Milestone 2: Proper RAG Evaluation Script

This script compares the TWO actual implementations built for the project:
1. Rule-based TF-IDF retrieval (from milestone2/rule_based/)
2. ML-based VerbatimRAG with SPLADE embeddings (from milestone2/ML_based/)

Outputs:
- Confusion Matrix Metrics: TP, FP, FN, TN, Precision, Recall, F1 Score
- Retrieval Metrics: Accuracy, Recall@k, MRR
- Visualization: Bar chart comparing Precision, Recall, F1 Score
- Side-by-side qualitative comparison
- CSV files:
  * confusion_matrix_metrics.csv - TP/FP/FN/TN and derived metrics
  * comparison_metrics.csv - Retrieval metrics
  * results_tfidf.csv - Detailed TF-IDF results
  * results_verbatimrag.csv - Detailed VerbatimRAG results
  * comparison_chart.png - Visualization (requires matplotlib)
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add ML_based to path for imports
ml_based_path = Path(__file__).parent.parent / "ML_based"
sys.path.insert(0, str(ml_based_path))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# VerbatimRAG imports
from verbatim_rag.document import Document, Chunk, ProcessedChunk, DocumentType, ChunkType
from verbatim_rag.ingestion import DocumentProcessor
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag import VerbatimIndex
from verbatim_rag.embedding_providers import SpladeProvider


# ============================================================================
# SYSTEM 1: RULE-BASED TF-IDF RETRIEVAL
# ============================================================================

class TFIDFRetriever:
    """Rule-based TF-IDF retrieval system (document-level)"""

    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_ids = [doc["id"] for doc in corpus]
        self.doc_titles = [doc.get("title", "") for doc in corpus]
        self.doc_texts = [doc["text"] for doc in corpus]

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        print(f"TF-IDF Retriever: Indexed {len(self.corpus)} documents")

    def retrieve(self, query, k=5):
        """Retrieve top-k documents for a query"""
        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        topk_idx = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            results.append({
                "rank": rank,
                "score": float(sims[idx]),
                "id": self.doc_ids[idx],
                "title": self.doc_titles[idx]
            })
        return results


# ============================================================================
# SYSTEM 2: ML-BASED VERBATIMRAG WITH SPLADE
# ============================================================================

class VerbatimRAGRetriever:
    """ML-based retrieval using VerbatimRAG with SPLADE embeddings"""

    def __init__(self, corpus, db_path):
        self.corpus = corpus
        self.db_path = db_path
        self.index = None
        self._build_index()

    def _build_index(self):
        """Build or load VerbatimRAG index"""
        import os

        # Prepare documents
        documents_for_index = []
        processor = DocumentProcessor()

        for paper in self.corpus:
            doc_obj = Document(
                title=paper['title'],
                source="json_corpus",
                content_type=DocumentType.TXT,
                raw_content=paper['text'],
                metadata={
                    "id": paper['id'],
                    "title": paper['title']
                }
            )

            # Chunk the text
            chunk_tuples = processor.chunker_provider.chunk(paper['text'])

            for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):
                enhanced_content = self._create_enhanced_content(struct_enhanced, doc_obj)

                doc_chunk = Chunk(
                    document_id=doc_obj.id,
                    content=raw_text,
                    chunk_number=i,
                    chunk_type=ChunkType.PARAGRAPH,
                )

                processed_chunk = ProcessedChunk(
                    chunk_id=doc_chunk.id,
                    enhanced_content=enhanced_content,
                )

                doc_chunk.add_processed_chunk(processed_chunk)
                doc_obj.add_chunk(doc_chunk)

            documents_for_index.append(doc_obj)

        # Setup vector store and index
        db_exists = os.path.exists(self.db_path)
        store = LocalMilvusStore(self.db_path, enable_sparse=True, enable_dense=False)

        sparse_embedder = SpladeProvider(
            model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
            device="cpu"
        )

        self.index = VerbatimIndex(vector_store=store, sparse_provider=sparse_embedder)

        if db_exists:
            try:
                res = store.client.query(store.collection_name, filter='id != ""', limit=1)
                if len(res) > 0:
                    print("VerbatimRAG: Using existing index")
                else:
                    print("VerbatimRAG: Database empty, indexing documents...")
                    self.index.add_documents(documents_for_index)
            except Exception as e:
                print(f"VerbatimRAG: Rebuilding index due to error: {e}")
                store.client.drop_collection(store.collection_name)
                self.index.add_documents(documents_for_index)
        else:
            print("VerbatimRAG: Creating new index...")
            self.index.add_documents(documents_for_index)

    def _create_enhanced_content(self, text, doc):
        """Create enhanced content with metadata"""
        parts = [text, "", "---"]
        parts.append(f"Document: {doc.title or 'Unknown'}")
        parts.append(f"Source: {doc.source or 'Unknown'}")
        for key, value in doc.metadata.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def retrieve(self, query, k=5):
        """Retrieve top-k documents for a query"""
        if not query.strip():
            return []

        # Query the index
        results = self.index.query(query, k=k)

        if not results:
            return []

        # Aggregate scores by paper ID
        paper_scores = defaultdict(float)
        paper_titles = {}

        for res in results:
            meta = getattr(res, 'metadata', {}) or {}
            if not meta and hasattr(res, 'get'):
                meta = res.get('metadata', {})

            paper_id = meta.get('id', 'Unknown')
            title = meta.get('title', 'Unknown')
            score = getattr(res, 'score', getattr(res, 'distance', 0.0))

            paper_scores[paper_id] += score
            paper_titles[paper_id] = title

        # Sort by total score
        sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k papers
        retrieved = []
        for rank, (paper_id, score) in enumerate(sorted_papers[:k], start=1):
            retrieved.append({
                "rank": rank,
                "score": float(score),
                "id": paper_id,
                "title": paper_titles.get(paper_id, "Unknown")
            })

        return retrieved


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def normalize_id(paper_id):
    """Normalize paper ID for comparison"""
    if paper_id is None:
        return None
    return str(paper_id).strip().lower()


def compute_confusion_matrix(results_df):
    """
    Compute confusion matrix elements for QA evaluation

    Only evaluates queries WITH ground truth answers.

    TP: Top-1 prediction matches ground truth
    FP: Top-1 prediction doesn't match ground truth (wrong answer)
    FN: System failed to answer (no retrieval or empty result)
    TN: Not applicable - only evaluating queries with ground truth
    """
    # Filter to only queries with ground truth
    valid_queries = results_df[results_df['correct_paper'].notna()].copy()

    # True Positives: Correct top-1 predictions
    tp = len(valid_queries[valid_queries['is_correct'] == True])

    # False Positives: Wrong top-1 predictions (predicted but incorrect)
    fp = len(valid_queries[valid_queries['is_correct'] == False])

    # False Negatives: Failed to retrieve (if predicted_paper is None/empty)
    # Systems always return results in current implementation, so FN = 0
    fn = len(valid_queries[valid_queries['predicted_paper'].isna()])

    # True Negatives: N/A - we only evaluate queries with ground truth
    tn = 0

    return tp, fp, fn, tn


def compute_precision_recall_f1(tp, fp, fn):
    """Compute Precision, Recall, and F1 Score from confusion matrix

    Standard formulas:
    - Precision = TP / (TP + FP) - of all predictions, how many were correct
    - Recall = TP / (TP + FN) - of all actual positives, how many were found
    - F1 = harmonic mean of precision and recall

    Note: When FN=0 (system always answers), Recall=1.0 is correct!
    This means the system attempted to answer all questions.
    Answer quality is measured by Precision.
    """
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_accuracy(results_df):
    """Compute accuracy on queries with ground truth"""
    valid_results = results_df[results_df["correct_paper"].notna()].copy()
    if len(valid_results) == 0:
        return 0.0
    return valid_results["is_correct"].mean()


def compute_recall_at_k(results_df, k):
    """Compute Recall@k on queries with ground truth"""
    valid_results = results_df[results_df["correct_paper"].notna()].copy()
    if len(valid_results) == 0:
        return 0.0

    recalls = []
    for _, row in valid_results.iterrows():
        correct_id = str(row["correct_paper"])
        retrieved_ids = [str(rid) for rid in row["retrieved_ids"][:k]]
        recalls.append(1 if correct_id in retrieved_ids else 0)

    return np.mean(recalls)


def compute_mrr(results_df):
    """Compute Mean Reciprocal Rank"""
    valid_results = results_df[results_df["correct_paper"].notna()].copy()
    if len(valid_results) == 0:
        return 0.0

    rrs = []
    for _, row in valid_results.iterrows():
        correct_id = str(row["correct_paper"])
        retrieved_ids = [str(rid) for rid in row["retrieved_ids"]]

        if correct_id in retrieved_ids:
            rank = retrieved_ids.index(correct_id) + 1
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)

    return np.mean(rrs)


# ============================================================================
# EVALUATION LOOP
# ============================================================================

def evaluate_system(retriever, test_queries, system_name):
    """Evaluate a retrieval system on test queries"""
    results = []

    for entry in test_queries:
        question = entry["question"]
        correct_paper = entry["correct_paper_id"]

        # Retrieve documents
        retrieved = retriever.retrieve(question, k=5)

        retrieved_ids = [r["id"] for r in retrieved]
        top_pred = retrieved[0]["id"] if retrieved else None

        # Use normalized comparison for correctness check
        is_correct = None
        if correct_paper is not None:
            is_correct = (normalize_id(correct_paper) == normalize_id(top_pred))

        results.append({
            "question": question,
            "predicted_paper": top_pred,
            "correct_paper": str(correct_paper) if correct_paper else None,
            "is_correct": is_correct,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [r["score"] for r in retrieved],
            "retrieved_titles": [r["title"] for r in retrieved]
        })

    df = pd.DataFrame(results)
    print(f"{system_name}: Evaluated {len(df)} queries")

    return df


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("=" * 80)
    print("MILESTONE 2: COMPARATIVE EVALUATION")
    print("TF-IDF (Rule-based) vs VerbatimRAG (ML-based)")
    print("=" * 80)

    # Load data
    base_path = Path(__file__).parent.parent.parent
    corpus_path = base_path / "corpus_json" / "corpus.json"
    query_path = base_path / "queries_json" / "queries.json"

    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    with query_path.open("r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"\nLoaded {len(corpus)} documents")
    print(f"Loaded {len(test_queries)} test queries")

    queries_with_gt = [q for q in test_queries if q["correct_paper_id"] is not None]
    queries_without_gt = [q for q in test_queries if q["correct_paper_id"] is None]
    print(f"Queries with ground truth: {len(queries_with_gt)}")
    print(f"Queries without ground truth: {len(queries_without_gt)}")
    print(f"\nNote: Metrics are computed only on the {len(queries_with_gt)} queries with ground truth.")

    # ========================================================================
    # SYSTEM 1: TF-IDF (Rule-based)
    # ========================================================================

    print("\n" + "-" * 80)
    print("SYSTEM 1: TF-IDF (Rule-based)")
    print("-" * 80)

    tfidf_retriever = TFIDFRetriever(corpus)
    df_tfidf = evaluate_system(tfidf_retriever, test_queries, "TF-IDF")

    # ========================================================================
    # SYSTEM 2: VerbatimRAG (ML-based)
    # ========================================================================

    print("\n" + "-" * 80)
    print("SYSTEM 2: VerbatimRAG with SPLADE (ML-based)")
    print("-" * 80)

    db_path = ml_based_path / "milvus_final.db"
    verbatim_retriever = VerbatimRAGRetriever(corpus, str(db_path))
    df_verbatim = evaluate_system(verbatim_retriever, test_queries, "VerbatimRAG")

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    # Compute confusion matrix for both systems
    tp_tfidf, fp_tfidf, fn_tfidf, tn_tfidf = compute_confusion_matrix(df_tfidf)
    precision_tfidf, recall_tfidf, f1_tfidf = compute_precision_recall_f1(tp_tfidf, fp_tfidf, fn_tfidf)

    tp_verbatim, fp_verbatim, fn_verbatim, tn_verbatim = compute_confusion_matrix(df_verbatim)
    precision_verbatim, recall_verbatim, f1_verbatim = compute_precision_recall_f1(tp_verbatim, fp_verbatim, fn_verbatim)

    print("\nConfusion Matrix - TF-IDF (Rule-based):")
    print(f"  TP: {tp_tfidf}, FP: {fp_tfidf}, FN: {fn_tfidf}, TN: {tn_tfidf} (N/A - only evaluating queries with ground truth)")
    print(f"  Precision: {precision_tfidf:.4f}, Recall: {recall_tfidf:.4f}, F1: {f1_tfidf:.4f}")

    print("\nConfusion Matrix - VerbatimRAG (ML-based):")
    print(f"  TP: {tp_verbatim}, FP: {fp_verbatim}, FN: {fn_verbatim}, TN: {tn_verbatim} (N/A - only evaluating queries with ground truth)")
    print(f"  Precision: {precision_verbatim:.4f}, Recall: {recall_verbatim:.4f}, F1: {f1_verbatim:.4f}")

    # Also compute retrieval metrics for comparison
    metrics_tfidf = {
        "Accuracy": compute_accuracy(df_tfidf),
        "Recall@1": compute_recall_at_k(df_tfidf, 1),
        "Recall@3": compute_recall_at_k(df_tfidf, 3),
        "Recall@5": compute_recall_at_k(df_tfidf, 5),
        "MRR": compute_mrr(df_tfidf)
    }

    metrics_verbatim = {
        "Accuracy": compute_accuracy(df_verbatim),
        "Recall@1": compute_recall_at_k(df_verbatim, 1),
        "Recall@3": compute_recall_at_k(df_verbatim, 3),
        "Recall@5": compute_recall_at_k(df_verbatim, 5),
        "MRR": compute_mrr(df_verbatim)
    }

    # Create comparison table for retrieval metrics
    comparison_data = {
        "System": ["TF-IDF (Rule-based)", "VerbatimRAG (ML-based)"],
        "Accuracy": [metrics_tfidf["Accuracy"], metrics_verbatim["Accuracy"]],
        "Recall@1": [metrics_tfidf["Recall@1"], metrics_verbatim["Recall@1"]],
        "Recall@3": [metrics_tfidf["Recall@3"], metrics_verbatim["Recall@3"]],
        "Recall@5": [metrics_tfidf["Recall@5"], metrics_verbatim["Recall@5"]],
        "MRR": [metrics_tfidf["MRR"], metrics_verbatim["MRR"]]
    }

    df_comparison = pd.DataFrame(comparison_data)

    # Create confusion matrix comparison table
    confusion_matrix_data = {
        "Metric": ["TP", "FP", "FN", "TN", "Precision", "Recall", "F1 Score"],
        "TF-IDF (Rule-based)": [
            tp_tfidf, fp_tfidf, fn_tfidf, tn_tfidf,
            precision_tfidf, recall_tfidf, f1_tfidf
        ],
        "VerbatimRAG (ML-based)": [
            tp_verbatim, fp_verbatim, fn_verbatim, tn_verbatim,
            precision_verbatim, recall_verbatim, f1_verbatim
        ]
    }

    df_confusion_matrix = pd.DataFrame(confusion_matrix_data)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_dir = Path(__file__).parent

    # Save all CSV files
    df_comparison.to_csv(output_dir / "comparison_metrics.csv", index=False)
    df_confusion_matrix.to_csv(output_dir / "confusion_matrix_metrics.csv", index=False)
    df_tfidf.to_csv(output_dir / "results_tfidf.csv", index=False)
    df_verbatim.to_csv(output_dir / "results_verbatimrag.csv", index=False)

    print("\nResults saved:")
    print(f"  {output_dir / 'comparison_metrics.csv'}")
    print(f"  {output_dir / 'confusion_matrix_metrics.csv'}")
    print(f"  {output_dir / 'results_tfidf.csv'}")
    print(f"  {output_dir / 'results_verbatimrag.csv'}")

    # Build qualitative comparison artifact covering every labeled query
    qualitative_rows = []
    labeled_indices = df_tfidf[df_tfidf["correct_paper"].notna()].index

    def format_prediction(row):
        """Return a string describing the top prediction with correctness marker."""
        pred = row["predicted_paper"] if pd.notna(row["predicted_paper"]) else "None"
        status = row["is_correct"]
        if status is True:
            marker = "✓"
        elif status is False:
            marker = "✗"
        else:
            marker = "N/A"
        return f"{pred} ({marker})"

    def locate_correct_rank(row):
        """Return 1-based rank of the ground-truth paper inside retrieved_ids."""
        correct_id = row["correct_paper"]
        retrieved_ids = row["retrieved_ids"]
        if correct_id is None or not isinstance(retrieved_ids, list):
            return None
        retrieved_strs = [str(rid) for rid in retrieved_ids]
        if correct_id in retrieved_strs:
            return retrieved_strs.index(correct_id) + 1
        return None

    for idx in labeled_indices:
        tfidf_row = df_tfidf.loc[idx]
        verbatim_row = df_verbatim.loc[idx]

        note_parts = []
        tfidf_rank = locate_correct_rank(tfidf_row)
        verbatim_rank = locate_correct_rank(verbatim_row)

        if tfidf_row["is_correct"] is False:
            if tfidf_rank:
                note_parts.append(f"TF-IDF correct at rank {tfidf_rank}")
            else:
                note_parts.append("TF-IDF missed in top-5")

        if verbatim_row["is_correct"] is False:
            if verbatim_rank:
                note_parts.append(f"VerbatimRAG correct at rank {verbatim_rank}")
            else:
                note_parts.append("VerbatimRAG missed in top-5")

        qualitative_rows.append({
            "Query": tfidf_row["question"],
            "Ground Truth": tfidf_row["correct_paper"],
            "TF-IDF Top-1": format_prediction(tfidf_row),
            "VerbatimRAG Top-1": format_prediction(verbatim_row),
            "Notes": "; ".join(note_parts)
        })

    df_qualitative = pd.DataFrame(qualitative_rows)
    qualitative_csv = output_dir / "qualitative_comparison.csv"
    df_qualitative.to_csv(qualitative_csv, index=False)

    try:
        qualitative_md = output_dir / "qualitative_comparison.md"
        md_header = "# Qualitative Comparison: TF-IDF vs VerbatimRAG\n\n"
        qualitative_md.write_text(md_header + df_qualitative.to_markdown(index=False) + "\n",
                                  encoding="utf-8")
        print(f"  {qualitative_csv}")
        print(f"  {qualitative_md}")
    except Exception as exc:
        print(f"  {qualitative_csv}")
        print(f"  Markdown export skipped: {exc}")

    # ========================================================================
    # GENERATE VISUALIZATION
    # ========================================================================

    try:
        import matplotlib.pyplot as plt

        print("\nGenerating visualization...")

        metrics = ['Precision', 'Recall', 'F1 Score']
        tfidf_values = [precision_tfidf, recall_tfidf, f1_tfidf]
        verbatim_values = [precision_verbatim, recall_verbatim, f1_verbatim]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, tfidf_values, width, label='TF-IDF (Rule-based)',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, verbatim_values, width, label='VerbatimRAG (ML-based)',
                       color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_title('Baseline System Comparison: Question-Answering Performance',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11, loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        viz_path = output_dir / 'comparison_chart.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"  {viz_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"\nVisualization generation failed: {e}")

    # ========================================================================
    # QUALITATIVE EXAMPLES
    # ========================================================================

    print("\n" + "=" * 80)
    print("QUALITATIVE EXAMPLES")
    print("=" * 80)

    for i in range(min(3, len(queries_with_gt))):
        query = queries_with_gt[i]
        question = query["question"]
        correct_paper = query["correct_paper_id"]
        correct_title = query.get("correct_paper_title", "")

        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"Query: {question}")
        print(f"Ground Truth: [{correct_paper}] {correct_title}")

        # Get results from both systems
        idx = test_queries.index(query)

        print(f"\n{'-'*80}")
        print("TF-IDF (Rule-based)")
        print(f"{'-'*80}")
        row = df_tfidf.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Top 3: {row['retrieved_ids'][:3]}")

        print(f"\n{'-'*80}")
        print("VerbatimRAG (ML-based)")
        print(f"{'-'*80}")
        row = df_verbatim.iloc[idx]
        print(f"  Predicted: {row['predicted_paper']}")
        print(f"  Correct: {'✓' if row['is_correct'] else '✗'}")
        print(f"  Top 3: {row['retrieved_ids'][:3]}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Display confusion matrix metrics
    print("\n--- CONFUSION MATRIX METRICS ---")
    df_cm_display = df_confusion_matrix.copy()
    # Format only the last 3 rows (Precision, Recall, F1)
    for idx in range(4, 7):
        df_cm_display.iloc[idx, 1] = f"{df_cm_display.iloc[idx, 1]:.4f} ({df_cm_display.iloc[idx, 1]*100:.2f}%)"
        df_cm_display.iloc[idx, 2] = f"{df_cm_display.iloc[idx, 2]:.4f} ({df_cm_display.iloc[idx, 2]*100:.2f}%)"

    print("\n" + df_cm_display.to_string(index=False))

    # Display retrieval metrics
    print("\n--- RETRIEVAL METRICS ---")
    df_display = df_comparison.copy()
    for col in df_display.columns:
        if col != "System":
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%")

    print("\n" + df_display.to_string(index=False))

    # Performance conclusion
    print("\n--- PERFORMANCE CONCLUSION ---")
    print(f"\nWinner: {'TF-IDF (Rule-based)' if f1_tfidf > f1_verbatim else 'VerbatimRAG (ML-based)'}")
    print(f"TF-IDF achieves {precision_tfidf*100:.2f}% precision vs VerbatimRAG's {precision_verbatim*100:.2f}%")
    print(f"F1 Score: TF-IDF {f1_tfidf*100:.2f}% vs VerbatimRAG {f1_verbatim*100:.2f}%")

    precision_diff = abs(precision_tfidf - precision_verbatim) * 100
    print(f"\nLargest difference is in Precision ({precision_diff:.2f}% gap)")

    if precision_tfidf > precision_verbatim:
        print("TF-IDF produces fewer false positives in document ranking.")
    else:
        print("VerbatimRAG produces fewer false positives in document ranking.")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return df_comparison, df_tfidf, df_verbatim


if __name__ == "__main__":
    main()
