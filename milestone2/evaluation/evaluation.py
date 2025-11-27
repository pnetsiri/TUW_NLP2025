"""
Milestone 2: Baseline Evaluation Script

This script evaluates all baseline implementations:
- Rule-based: TF-IDF (document-level and chunk-level)
- ML-based: SPLADE sparse encoder with Milvus vector store

Outputs:
- Quantitative metrics comparison table (including latency measurements)
- Qualitative analysis with side-by-side comparisons
- Failure case analysis
- System performance metrics (latency per query)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import time
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# HELPER FUNCTIONS FOR EVALUATION METRICS
# ============================================================================


def recall_at_k(retrieved_ids, correct_id, k):
    """Check if correct document is in top-k retrieved documents."""
    if correct_id is None or correct_id == "None":
        return None
    return 1 if str(correct_id) in [str(id) for id in retrieved_ids[:k]] else 0


def mean_reciprocal_rank(retrieved_ids, correct_id):
    """Calculate reciprocal rank of the correct document."""
    if correct_id is None or correct_id == "None":
        return None

    correct_id_str = str(correct_id)
    retrieved_ids_str = [str(id) for id in retrieved_ids]

    if correct_id_str in retrieved_ids_str:
        rank = retrieved_ids_str.index(correct_id_str) + 1
        return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids, correct_id, k):
    """Calculate Discounted Cumulative Gain at k."""
    if correct_id is None or correct_id == "None":
        return None

    correct_id_str = str(correct_id)
    retrieved_ids_str = [str(id) for id in retrieved_ids[:k]]

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids_str):
        if doc_id == correct_id_str:
            dcg += 1.0 / np.log2(i + 2)
            break
    return dcg


def ndcg_at_k(retrieved_ids, correct_id, k):
    """Calculate Normalized Discounted Cumulative Gain at k."""
    dcg = dcg_at_k(retrieved_ids, correct_id, k)
    if dcg is None:
        return None

    # Ideal DCG (correct doc at position 0)
    idcg = 1.0 / np.log2(2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results_df):
    """Compute all retrieval metrics from results dataframe."""
    # Filter only queries with ground truth
    valid_results = results_df[results_df["correct_paper"].notna()].copy()

    if len(valid_results) == 0:
        return {}

    metrics = {}

    # Accuracy (top-1)
    metrics["Accuracy"] = valid_results["is_correct"].mean()

    # Recall@k
    for k in [1, 3, 5]:
        recall_k_values = []
        for _, row in valid_results.iterrows():
            r = recall_at_k(row["retrieved_ids"], row["correct_paper"], k)
            if r is not None:
                recall_k_values.append(r)
        metrics[f"Recall@{k}"] = np.mean(recall_k_values) if recall_k_values else 0.0

    # MRR
    mrr_values = []
    for _, row in valid_results.iterrows():
        mrr = mean_reciprocal_rank(row["retrieved_ids"], row["correct_paper"])
        if mrr is not None:
            mrr_values.append(mrr)
    metrics["MRR"] = np.mean(mrr_values) if mrr_values else 0.0

    # nDCG@k
    for k in [3, 5]:
        ndcg_values = []
        for _, row in valid_results.iterrows():
            ndcg = ndcg_at_k(row["retrieved_ids"], row["correct_paper"], k)
            if ndcg is not None:
                ndcg_values.append(ndcg)
        metrics[f"nDCG@{k}"] = np.mean(ndcg_values) if ndcg_values else 0.0

    # Precision and Recall (treating it as binary classification)
    y_true = (
        valid_results["correct_paper"] == valid_results["predicted_paper"]
    ).astype(int)
    y_pred = valid_results["is_correct"].astype(int)

    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def main():
    print("=" * 100)
    print("MILESTONE 2: BASELINE EVALUATION")
    print("=" * 100)

    # Load data - UPDATED PATH: now go up two levels from evaluation/ folder
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
    print(f"Queries with ground truth: {len(queries_with_gt)}")

    # ========================================================================
    # BASELINE 1: TF-IDF DOCUMENT-LEVEL RETRIEVAL
    # ========================================================================

    print("\n" + "-" * 100)
    print("EVALUATING BASELINE 1: TF-IDF DOCUMENT-LEVEL RETRIEVAL")
    print("-" * 100)

    doc_ids = [doc["id"] for doc in corpus]
    doc_titles = [doc.get("title", "") for doc in corpus]
    doc_texts = [doc["text"] for doc in corpus]

    vectorizer_doc = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1
    )

    tfidf_matrix_doc = vectorizer_doc.fit_transform(doc_texts)
    print(f"Built TF-IDF matrix: {tfidf_matrix_doc.shape}")

    def retrieve_tfidf_documents(query: str, k: int = 5):
        if not query.strip():
            return []

        q_vec = vectorizer_doc.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix_doc)[0]
        topk_idx = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": float(sims[idx]),
                    "id": doc_ids[idx],
                    "title": doc_titles[idx],
                    "text": doc_texts[idx],
                }
            )
        return results

    tfidf_doc_results = []
    tfidf_doc_latencies = []

    for entry in test_queries:
        question = entry["question"]
        correct_paper = entry["correct_paper_id"]

        # Measure retrieval latency
        start_time = time.time()
        retrieved = retrieve_tfidf_documents(query=question, k=5)
        latency = time.time() - start_time
        tfidf_doc_latencies.append(latency)

        retrieved_ids = [r["id"] for r in retrieved]

        top_pred = retrieved[0]["id"] if retrieved else None

        tfidf_doc_results.append(
            {
                "question": question,
                "predicted_paper": top_pred,
                "correct_paper": str(correct_paper) if correct_paper else None,
                "is_correct": (
                    str(correct_paper) == str(top_pred) if correct_paper else None
                ),
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [r["score"] for r in retrieved],
                "retrieved_titles": [r["title"] for r in retrieved],
                "latency_ms": latency * 1000,  # Convert to milliseconds
            }
        )

    df_tfidf_doc = pd.DataFrame(tfidf_doc_results)
    print(f"Evaluated {len(df_tfidf_doc)} queries")
    print(f"Average latency: {np.mean(tfidf_doc_latencies)*1000:.2f} ms (min: {np.min(tfidf_doc_latencies)*1000:.2f} ms, max: {np.max(tfidf_doc_latencies)*1000:.2f} ms)")

    # ========================================================================
    # BASELINE 2: TF-IDF CHUNK-LEVEL RETRIEVAL
    # ========================================================================

    print("\n" + "-" * 100)
    print("EVALUATING BASELINE 2: TF-IDF CHUNK-LEVEL RETRIEVAL")
    print("-" * 100)

    def chunk_text(text, chunk_size=220, overlap=40):
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        n = len(words)

        while start < n:
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            if end >= n:
                break

            start = end - overlap

        return chunks

    passage_texts = []
    passage_meta = []

    for doc in corpus:
        doc_id = doc["id"]
        title = doc.get("title", "")
        text = doc["text"]

        chunks = chunk_text(text, chunk_size=220, overlap=40)
        start_word = 0

        for i, chunk in enumerate(chunks):
            end_word = start_word + len(chunk.split())
            passage_texts.append(chunk)
            passage_meta.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "start_word": start_word,
                    "end_word": end_word,
                }
            )
            start_word = end_word - 40

    print(f"Created {len(passage_texts)} chunks from {len(corpus)} documents")

    vectorizer_chunk = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1
    )

    tfidf_matrix_chunk = vectorizer_chunk.fit_transform(passage_texts)
    print(f"Built TF-IDF matrix: {tfidf_matrix_chunk.shape}")

    def retrieve_tfidf_chunks(query: str, k: int = 5):
        if not query.strip():
            return []

        q_vec = vectorizer_chunk.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix_chunk)[0]
        topk_idx = np.argsort(sims)[::-1][:k]

        results = []
        for rank, idx in enumerate(topk_idx, start=1):
            meta = passage_meta[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(sims[idx]),
                    "text": passage_texts[idx],
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "chunk_id": meta["chunk_id"],
                }
            )
        return results

    tfidf_chunk_results = []
    tfidf_chunk_latencies = []

    for entry in test_queries:
        question = entry["question"]
        correct_paper = entry["correct_paper_id"]

        # Measure retrieval latency
        start_time = time.time()
        retrieved = retrieve_tfidf_chunks(query=question, k=5)
        latency = time.time() - start_time
        tfidf_chunk_latencies.append(latency)

        retrieved_ids = [r["doc_id"] for r in retrieved]

        top_pred = retrieved[0]["doc_id"] if retrieved else None

        tfidf_chunk_results.append(
            {
                "question": question,
                "predicted_paper": top_pred,
                "correct_paper": str(correct_paper) if correct_paper else None,
                "is_correct": (
                    str(correct_paper) == str(top_pred) if correct_paper else None
                ),
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [r["score"] for r in retrieved],
                "retrieved_chunks": [r["text"][:200] for r in retrieved],
                "latency_ms": latency * 1000,  # Convert to milliseconds
            }
        )

    df_tfidf_chunk = pd.DataFrame(tfidf_chunk_results)
    print(f"Evaluated {len(df_tfidf_chunk)} queries")
    print(f"Average latency: {np.mean(tfidf_chunk_latencies)*1000:.2f} ms (min: {np.min(tfidf_chunk_latencies)*1000:.2f} ms, max: {np.max(tfidf_chunk_latencies)*1000:.2f} ms)")

    # ========================================================================
    # BASELINE 3: ML-BASED SPLADE RETRIEVAL
    # ========================================================================

    print("\n" + "-" * 100)
    print("EVALUATING BASELINE 3: ML-BASED SPLADE RETRIEVAL")
    print("-" * 100)

    # UPDATED PATH: go up one level to milestone2, then into ML_based
    ml_based_path = Path(__file__).parent.parent / "ML_based"
    sys.path.insert(0, str(ml_based_path))

    try:
        from ML_based_classification import find_best_paper, index

        ml_available = True
        print("ML-based SPLADE model loaded successfully")
    except Exception as e:
        print(f"✗ Could not load ML-based model: {e}")
        print("Skipping ML-based evaluation")
        ml_available = False
        df_splade = None

    if ml_available:
        splade_results = []
        splade_latencies = []

        for entry in test_queries:
            question = entry["question"]
            correct_paper = entry["correct_paper_id"]

            # Measure retrieval latency
            start_time = time.time()
            results = index.query(question, k=5)
            latency = time.time() - start_time
            splade_latencies.append(latency)

            retrieved_ids = []
            retrieved_scores = []

            for res in results:
                meta = getattr(res, "metadata", {}) or {}
                if not meta and hasattr(res, "get"):
                    meta = res.get("metadata", {})
                doc_id = meta.get("id", "Unknown")
                retrieved_ids.append(doc_id)

                score = getattr(res, "score", None)
                if score is None:
                    score = getattr(res, "distance", 0.0)
                retrieved_scores.append(score)

            top_pred = retrieved_ids[0] if retrieved_ids else None

            splade_results.append(
                {
                    "question": question,
                    "predicted_paper": top_pred,
                    "correct_paper": str(correct_paper) if correct_paper else None,
                    "is_correct": (
                        str(correct_paper) == str(top_pred) if correct_paper else None
                    ),
                    "retrieved_ids": retrieved_ids,
                    "retrieved_scores": retrieved_scores,
                    "latency_ms": latency * 1000,  # Convert to milliseconds
                }
            )

        df_splade = pd.DataFrame(splade_results)
        print(f"Evaluated {len(df_splade)} queries")
        print(f"Average latency: {np.mean(splade_latencies)*1000:.2f} ms (min: {np.min(splade_latencies)*1000:.2f} ms, max: {np.max(splade_latencies)*1000:.2f} ms)")

    # ========================================================================
    # COMPUTE QUANTITATIVE METRICS
    # ========================================================================

    print("\n" + "=" * 100)
    print("QUANTITATIVE METRICS COMPARISON")
    print("=" * 100)

    metrics_tfidf_doc = compute_retrieval_metrics(df_tfidf_doc)
    metrics_tfidf_chunk = compute_retrieval_metrics(df_tfidf_chunk)

    # Add latency metrics
    metrics_tfidf_doc["Avg_Latency_ms"] = np.mean(tfidf_doc_latencies) * 1000
    metrics_tfidf_chunk["Avg_Latency_ms"] = np.mean(tfidf_chunk_latencies) * 1000

    comparison_data = {
        "Baseline": ["TF-IDF (Document)", "TF-IDF (Chunk)"],
        **{
            metric: [
                metrics_tfidf_doc.get(metric, 0),
                metrics_tfidf_chunk.get(metric, 0),
            ]
            for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Recall@1",
                "Recall@3",
                "Recall@5",
                "MRR",
                "nDCG@3",
                "nDCG@5",
                "Avg_Latency_ms",
            ]
        },
    }

    if ml_available and df_splade is not None:
        metrics_splade = compute_retrieval_metrics(df_splade)
        metrics_splade["Avg_Latency_ms"] = np.mean(splade_latencies) * 1000
        comparison_data["Baseline"].append("SPLADE (ML-based)")
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "MRR",
            "nDCG@3",
            "nDCG@5",
            "Avg_Latency_ms",
        ]:
            comparison_data[metric].append(metrics_splade.get(metric, 0))

    df_comparison = pd.DataFrame(comparison_data)

    # Format as percentages (except latency which is in ms)
    df_comparison_display = df_comparison.copy()
    for col in df_comparison_display.columns:
        if col == "Baseline":
            continue
        elif col == "Avg_Latency_ms":
            df_comparison_display[col] = df_comparison_display[col].apply(
                lambda x: f"{x:.2f} ms"
            )
        else:
            df_comparison_display[col] = df_comparison_display[col].apply(
                lambda x: f"{x*100:.2f}%"
            )

    print("\n" + df_comparison_display.to_string(index=False))
    print("=" * 100)

    # Save results - OUTPUT TO CURRENT DIRECTORY (evaluation/)
    output_dir = Path(__file__).parent
    df_comparison.to_csv(output_dir / "evaluation_metrics_comparison.csv", index=False)
    df_tfidf_doc.to_csv(output_dir / "results_tfidf_document.csv", index=False)
    df_tfidf_chunk.to_csv(output_dir / "results_tfidf_chunk.csv", index=False)
    if ml_available and df_splade is not None:
        df_splade.to_csv(output_dir / "results_splade_ml.csv", index=False)

    print("\nMetrics saved to:")
    print(f"  {output_dir / 'evaluation_metrics_comparison.csv'}")
    print(f"  {output_dir / 'results_tfidf_document.csv'}")
    print(f"  {output_dir / 'results_tfidf_chunk.csv'}")
    if ml_available:
        print(f"  {output_dir / 'results_splade_ml.csv'}")

    # ========================================================================
    # QUALITATIVE ANALYSIS
    # ========================================================================

    print("\n" + "=" * 100)
    print("QUALITATIVE ANALYSIS: EXAMPLE RETRIEVALS")
    print("=" * 100)

    # Show 3 example comparisons
    queries_with_gt_indices = [
        i for i, q in enumerate(test_queries) if q["correct_paper_id"] is not None
    ]

    for example_num, idx in enumerate(queries_with_gt_indices[:3], 1):
        query_entry = test_queries[idx]
        question = query_entry["question"]
        correct_paper = query_entry["correct_paper_id"]
        correct_title = query_entry["correct_paper_title"]

        print(f"\n{'='*100}")
        print(f"EXAMPLE {example_num}")
        print(f"{'='*100}")
        print(f"\nQuery: {question}")
        print(f"\nGround Truth: [{correct_paper}] {correct_title}")

        # TF-IDF Document
        tfidf_doc_row = df_tfidf_doc.iloc[idx]
        print(f"\nTF-IDF (Document-Level)")
        print(f"   Predicted: {tfidf_doc_row['predicted_paper']}")
        print(f"   Correct: {'PASS' if tfidf_doc_row['is_correct'] else 'FAIL'}")
        print(f"   Top-3: {tfidf_doc_row['retrieved_ids'][:3]}")

        # TF-IDF Chunk
        tfidf_chunk_row = df_tfidf_chunk.iloc[idx]
        print(f"\nTF-IDF (Chunk-Level)")
        print(f"   Predicted: {tfidf_chunk_row['predicted_paper']}")
        print(f"   Correct: {'PASS' if tfidf_chunk_row['is_correct'] else 'FAIL'}")
        print(f"   Top-3: {tfidf_chunk_row['retrieved_ids'][:3]}")

        # SPLADE
        if ml_available and df_splade is not None:
            splade_row = df_splade.iloc[idx]
            print(f"\nSPLADE (ML-based)")
            print(f"   Predicted: {splade_row['predicted_paper']}")
            print(f"   Correct: {'PASS' if splade_row['is_correct'] else 'FAIL'}")
            print(f"   Top-3: {splade_row['retrieved_ids'][:3]}")

    # ========================================================================
    # FAILURE ANALYSIS
    # ========================================================================

    print("\n" + "=" * 100)
    print("FAILURE CASE ANALYSIS")
    print("=" * 100)

    all_failed = []
    partial_success = []

    for idx in queries_with_gt_indices:
        tfidf_doc_correct = df_tfidf_doc.iloc[idx]["is_correct"]
        tfidf_chunk_correct = df_tfidf_chunk.iloc[idx]["is_correct"]

        if ml_available and df_splade is not None:
            splade_correct = df_splade.iloc[idx]["is_correct"]
            if not (tfidf_doc_correct or tfidf_chunk_correct or splade_correct):
                all_failed.append(idx)
            elif tfidf_doc_correct and tfidf_chunk_correct and splade_correct:
                pass  # all succeeded
            else:
                partial_success.append(idx)
        else:
            if not (tfidf_doc_correct or tfidf_chunk_correct):
                all_failed.append(idx)
            elif tfidf_doc_correct != tfidf_chunk_correct:
                partial_success.append(idx)

    print(f"\nQueries where ALL methods failed: {len(all_failed)}")
    print(f"Queries with MIXED results: {len(partial_success)}")

    if all_failed:
        print("\n" + "-" * 100)
        print("COMPLETE FAILURE EXAMPLES:")
        print("-" * 100)
        for idx in all_failed[:2]:
            query = test_queries[idx]
            print(f"\nQuery: {query['question']}")
            print(f"   Ground Truth: [{query['correct_paper_id']}]")
            print(
                f"   TF-IDF Doc predicted: {df_tfidf_doc.iloc[idx]['predicted_paper']}"
            )
            print(
                f"   TF-IDF Chunk predicted: {df_tfidf_chunk.iloc[idx]['predicted_paper']}"
            )
            if ml_available and df_splade is not None:
                print(f"   SPLADE predicted: {df_splade.iloc[idx]['predicted_paper']}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "#" * 100)
    print("EVALUATION SUMMARY")
    print("#" * 100)

    print(f"\nKey Findings:")
    print(f"\n   TF-IDF (Document-Level):")
    print(f"      • Accuracy: {metrics_tfidf_doc['Accuracy']*100:.2f}%")
    print(f"      • Recall@3: {metrics_tfidf_doc['Recall@3']*100:.2f}%")
    print(f"      • MRR: {metrics_tfidf_doc['MRR']:.4f}")
    print(f"      • Avg Latency: {metrics_tfidf_doc['Avg_Latency_ms']:.2f} ms")

    print(f"\n   TF-IDF (Chunk-Level):")
    print(f"      • Accuracy: {metrics_tfidf_chunk['Accuracy']*100:.2f}%")
    print(f"      • Recall@3: {metrics_tfidf_chunk['Recall@3']*100:.2f}%")
    print(f"      • MRR: {metrics_tfidf_chunk['MRR']:.4f}")
    print(f"      • Avg Latency: {metrics_tfidf_chunk['Avg_Latency_ms']:.2f} ms")

    if ml_available and df_splade is not None:
        print(f"\n   SPLADE (ML-based):")
        print(f"      • Accuracy: {metrics_splade['Accuracy']*100:.2f}%")
        print(f"      • Recall@3: {metrics_splade['Recall@3']*100:.2f}%")
        print(f"      • MRR: {metrics_splade['MRR']:.4f}")
        print(f"      • Avg Latency: {metrics_splade['Avg_Latency_ms']:.2f} ms")

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)

    return (
        df_comparison,
        df_tfidf_doc,
        df_tfidf_chunk,
        df_splade if ml_available else None,
    )


if __name__ == "__main__":
    main()
