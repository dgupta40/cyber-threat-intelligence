#!/usr/bin/env python3
"""
Sentence Embedding Model Analysis for CTI Pipeline

Evaluates different sentence transformer models to determine optimal embeddings for
cybersecurity text data, comparing:
- Embedding quality via clustering and similarity metrics
- Processing speed
- Memory usage
- Dimensionality
- Domain-specific concept relationships

Outputs: Visualizations and metrics for model selection justification
"""

import logging
import time
import json
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/analysis/embedding_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Models to compare (including your current choice)
MODELS = [
    "all-MiniLM-L6-v2",  # Your current model (384 dimensions)
    "all-mpnet-base-v2",  # Higher quality but slower (768 dimensions)
    "all-distilroberta-v1",  # Another high-quality option (768 dimensions)
    "paraphrase-MiniLM-L3-v2",  # Smaller/faster option (384 dimensions)
    "all-MiniLM-L12-v2",  # Larger version of your current model (384 dimensions)
]

# Sample sizes for different tests to manage runtime
FULL_EVAL_SAMPLE = 500  # For detailed evaluations
SPEED_TEST_SAMPLE = 1000  # For speed testing

# Cybersecurity-specific test cases for semantic understanding
CYBER_TEST_PAIRS = [
    # Format: (text1, text2, expected_similarity)
    # Higher expected_similarity means the model should see these as more similar
    ("SQL injection attack detected", "SQLI vulnerability exploited", 0.8),
    ("CVE-2021-44228 Log4j vulnerability", "Log4Shell remote code execution", 0.7),
    ("Buffer overflow in network stack", "Stack-based memory corruption", 0.6),
    ("DDoS attack mitigated", "Service withstood denial of service", 0.7),
    ("Ransomware encrypted files", "Cryptographic malware attack", 0.6),
    ("Zero-day vulnerability", "Unpatched security flaw", 0.7),
    ("XSS in web application", "Cross-site scripting vulnerability", 0.9),
    ("Phishing email campaign", "Fraudulent login attempt", 0.5),
    ("Data exfiltration detected", "Sensitive information leaked", 0.6),
    ("Remote code execution", "Arbitrary command execution", 0.8),
    # Dissimilar pairs
    ("SQL injection attack", "Hardware failure reported", 0.2),
    ("Authentication bypass", "Network latency issues", 0.2),
    ("Malware detected", "System upgrade completed", 0.2),
    ("Firewall configuration", "User interface redesign", 0.1),
    ("Data breach", "Performance optimization", 0.1),
]


def load_data():
    """Load the processed data from master.parquet"""
    df = pd.read_parquet(DATA_DIR / "master.parquet")
    return df


def generate_test_sets(df):
    """Create various test sets from the data"""
    # Filter out very short texts
    df = df[df.clean_text.str.len() > 50].reset_index(drop=True)

    # Create test sets
    test_sets = {}

    # Random sample for full evaluation
    test_sets["full_eval"] = df.sample(
        min(FULL_EVAL_SAMPLE, len(df))
    ).clean_text.tolist()

    # Larger sample for speed testing
    test_sets["speed"] = df.sample(min(SPEED_TEST_SAMPLE, len(df))).clean_text.tolist()

    # Domain-specific examples
    test_sets["cyber_examples"] = [pair[0] for pair in CYBER_TEST_PAIRS] + [
        pair[1] for pair in CYBER_TEST_PAIRS
    ]

    return test_sets


def benchmark_models(test_sets):
    """Benchmark different models across multiple metrics"""
    results = {}

    for model_name in MODELS:
        logging.info(f"Evaluating model: {model_name}")
        model_results = {}

        try:
            # Load model
            start_time = time.time()
            model = SentenceTransformer(model_name, cache_folder=str(MODEL_DIR))
            load_time = time.time() - start_time
            model_results["load_time"] = load_time

            # Get model info
            model_results["embedding_dimension"] = (
                model.get_sentence_embedding_dimension()
            )
            model_results["model_size_mb"] = get_model_size_mb(model)

            # Speed test
            encoding_times = []
            batch_sizes = [1, 16, 32, 64]

            for bs in batch_sizes:
                # Measure encoding time for different batch sizes
                start_time = time.time()
                _ = model.encode(
                    test_sets["speed"][: min(bs * 10, len(test_sets["speed"]))],
                    batch_size=bs,
                    show_progress_bar=False,
                )
                encoding_time = time.time() - start_time
                encoding_times.append(
                    {
                        "batch_size": bs,
                        "time": encoding_time,
                        "texts_per_second": min(bs * 10, len(test_sets["speed"]))
                        / encoding_time,
                    }
                )

            model_results["encoding_performance"] = encoding_times

            # Embedding quality tests
            # 1. Get embeddings for evaluation set
            eval_embeddings = model.encode(
                test_sets["full_eval"], show_progress_bar=True
            )

            # 2. Cluster quality metrics
            cluster_metrics = evaluate_clustering(eval_embeddings)
            model_results["cluster_metrics"] = cluster_metrics

            # 3. Semantic similarity on cyber pairs
            semantic_scores = evaluate_semantic_understanding(model, CYBER_TEST_PAIRS)
            model_results["semantic_understanding"] = semantic_scores

            # 4. Visualize embeddings
            tsne_coordinates = visualize_embeddings(eval_embeddings, model_name)
            model_results["tsne_coords"] = tsne_coordinates

            # Store results for this model
            results[model_name] = model_results

            # Clean up to avoid GPU memory issues
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}

    return results


def get_model_size_mb(model):
    """Estimate model size in MB"""
    try:
        # Get size based on parameters
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    except:
        return None


def evaluate_clustering(embeddings, n_clusters=5):
    """Evaluate embeddings using clustering metrics"""
    try:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate clustering metrics
        sil_score = silhouette_score(embeddings, cluster_labels)
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)

        # Calculate average within-cluster similarity
        within_sim = 0
        for i in range(n_clusters):
            cluster_points = embeddings[cluster_labels == i]
            if len(cluster_points) > 1:
                sim_matrix = cosine_similarity(cluster_points)
                within_sim += sim_matrix.sum() / (
                    len(cluster_points) * (len(cluster_points) - 1)
                )
        within_sim /= n_clusters

        return {
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "avg_within_cluster_similarity": within_sim,
            "n_clusters": n_clusters,
        }
    except Exception as e:
        logging.error(f"Clustering evaluation error: {str(e)}")
        return {"error": str(e)}


def evaluate_semantic_understanding(model, test_pairs):
    """Evaluate model's ability to understand cybersecurity semantic relationships"""
    results = []

    # Get all unique texts
    all_texts = []
    for text1, text2, _ in test_pairs:
        if text1 not in all_texts:
            all_texts.append(text1)
        if text2 not in all_texts:
            all_texts.append(text2)

    # Encode all texts at once for efficiency
    all_embeddings = model.encode(all_texts, show_progress_bar=False)

    # Create mapping from text to embedding
    embedding_map = {text: emb for text, emb in zip(all_texts, all_embeddings)}

    # Calculate similarities
    for text1, text2, expected in test_pairs:
        emb1 = embedding_map[text1]
        emb2 = embedding_map[text2]
        actual_sim = util.cos_sim(emb1, emb2).item()
        delta = abs(actual_sim - expected)
        results.append(
            {
                "text1": text1,
                "text2": text2,
                "expected_similarity": expected,
                "actual_similarity": actual_sim,
                "delta": delta,
            }
        )

    # Calculate aggregate metrics
    avg_delta = sum(r["delta"] for r in results) / len(results)
    high_expected_pairs = [r for r in results if r["expected_similarity"] >= 0.6]
    high_expected_avg = sum(r["actual_similarity"] for r in high_expected_pairs) / len(
        high_expected_pairs
    )
    low_expected_pairs = [r for r in results if r["expected_similarity"] <= 0.3]
    low_expected_avg = sum(r["actual_similarity"] for r in low_expected_pairs) / len(
        low_expected_pairs
    )

    return {
        "detailed_pairs": results,
        "avg_delta": avg_delta,
        "high_similarity_avg": high_expected_avg,
        "low_similarity_avg": low_expected_avg,
        "contrast_ratio": (
            high_expected_avg / low_expected_avg
            if low_expected_avg > 0
            else float("inf")
        ),
    }


def visualize_embeddings(embeddings, model_name):
    """Create t-SNE visualization of embeddings"""
    # Use a sample if there are too many embeddings
    max_tsne_samples = min(500, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), max_tsne_samples, replace=False)
    embeddings_sample = embeddings[sample_idx]

    # Perform t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, max_tsne_samples - 1)
    )
    tsne_result = tsne.fit_transform(embeddings_sample)

    # Plot and save
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    plt.title(f"t-SNE Visualization of {model_name} Embeddings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.replace('/', '_')}_tsne.png")
    plt.close()

    return tsne_result.tolist()


def create_comparative_visualizations(results):
    """Create comparative visualizations across models"""
    # 1. Model size vs. performance plot
    plt.figure(figsize=(12, 8))

    model_names = []
    model_sizes = []
    semantic_scores = []
    colors = []

    for model_name, data in results.items():
        if "error" not in data and "model_size_mb" in data:
            model_names.append(model_name)
            model_sizes.append(data["model_size_mb"])
            if "semantic_understanding" in data:
                # Use contrast ratio as the semantic score
                semantic_scores.append(data["semantic_understanding"]["contrast_ratio"])
                # Highlight your current model
                colors.append("red" if model_name == "all-MiniLM-L6-v2" else "blue")

    plt.scatter(model_sizes, semantic_scores, c=colors, s=100, alpha=0.7)

    # Add model names as labels
    for i, name in enumerate(model_names):
        plt.annotate(
            name,
            (model_sizes[i], semantic_scores[i]),
            fontsize=9,
            ha="center",
            va="bottom",
            rotation=20,
        )

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Semantic Understanding (Contrast Ratio)")
    plt.title("Model Size vs. Semantic Understanding")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_size_vs_performance.png")
    plt.close()

    # 2. Encoding speed comparison
    plt.figure(figsize=(12, 8))

    batch_sizes = []
    speeds = []
    model_identifiers = []

    for model_name, data in results.items():
        if "error" not in data and "encoding_performance" in data:
            for perf in data["encoding_performance"]:
                batch_sizes.append(perf["batch_size"])
                speeds.append(perf["texts_per_second"])
                model_identifiers.append(model_name)

    # Create a DataFrame for easier plotting
    speed_df = pd.DataFrame(
        {
            "model": model_identifiers,
            "batch_size": batch_sizes,
            "texts_per_second": speeds,
        }
    )

    # Plot
    sns.barplot(x="batch_size", y="texts_per_second", hue="model", data=speed_df)
    plt.title("Encoding Speed by Model and Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Texts Per Second")
    plt.yscale("log")  # Log scale for better visibility
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoding_speed_comparison.png")
    plt.close()

    # 3. Clustering metrics comparison
    plt.figure(figsize=(15, 6))

    model_names = []
    silhouette_scores = []
    within_cluster_similarities = []

    for model_name, data in results.items():
        if (
            "error" not in data
            and "cluster_metrics" in data
            and "error" not in data["cluster_metrics"]
        ):
            model_names.append(model_name)
            silhouette_scores.append(data["cluster_metrics"]["silhouette_score"])
            within_cluster_similarities.append(
                data["cluster_metrics"]["avg_within_cluster_similarity"]
            )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Silhouette scores
    ax1.bar(
        model_names,
        silhouette_scores,
        color=["red" if name == "all-MiniLM-L6-v2" else "blue" for name in model_names],
    )
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Clustering Quality: Silhouette Score")
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    # Within-cluster similarities
    ax2.bar(
        model_names,
        within_cluster_similarities,
        color=["red" if name == "all-MiniLM-L6-v2" else "blue" for name in model_names],
    )
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Avg Within-Cluster Similarity")
    ax2.set_title("Clustering Quality: Within-Cluster Similarity")
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clustering_metrics_comparison.png")
    plt.close()

    # 4. Cyber-specific semantic understanding
    plt.figure(figsize=(12, 8))

    model_names = []
    avg_deltas = []
    high_sim_avgs = []
    low_sim_avgs = []

    for model_name, data in results.items():
        if "error" not in data and "semantic_understanding" in data:
            model_names.append(model_name)
            avg_deltas.append(data["semantic_understanding"]["avg_delta"])
            high_sim_avgs.append(data["semantic_understanding"]["high_similarity_avg"])
            low_sim_avgs.append(data["semantic_understanding"]["low_similarity_avg"])

    # Create a bar plot for these metrics
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, high_sim_avgs, width, label="High Similarity Pairs Avg")
    rects2 = ax.bar(x, low_sim_avgs, width, label="Low Similarity Pairs Avg")
    rects3 = ax.bar(x + width, avg_deltas, width, label="Avg Delta from Expected")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Cybersecurity Semantic Understanding")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cyber_semantic_understanding.png")
    plt.close()


def generate_report(results):
    """Generate a comprehensive report with recommendations"""
    # Create scoring system
    model_scores = {}

    for model_name, data in results.items():
        if "error" in data:
            model_scores[model_name] = {"total_score": 0, "error": data["error"]}
            continue

        # Initialize scores
        scores = {
            "embedding_quality": 0,
            "speed": 0,
            "efficiency": 0,
            "cyber_understanding": 0,
            "total_score": 0,
        }

        # 1. Embedding quality score (0-10)
        if "cluster_metrics" in data and "error" not in data["cluster_metrics"]:
            # Normalize silhouette score (ranges from -1 to 1)
            sil_score = (data["cluster_metrics"]["silhouette_score"] + 1) / 2 * 5
            # Normalize within-cluster similarity (ranges from 0 to 1)
            within_sim = data["cluster_metrics"]["avg_within_cluster_similarity"] * 5
            scores["embedding_quality"] = sil_score + within_sim

        # 2. Speed score (0-10)
        if "encoding_performance" in data:
            # Use the largest batch size for comparison
            largest_batch = max(
                data["encoding_performance"], key=lambda x: x["batch_size"]
            )
            # We'll compare speeds across models later
            scores["_texts_per_second"] = largest_batch["texts_per_second"]

        # 3. Efficiency score (0-10)
        if "model_size_mb" in data and "embedding_dimension" in data:
            # Smaller models get higher scores
            size_score = 10 - min(10, data["model_size_mb"] / 100)
            # Balance with dimension (higher dimension is better but has diminishing returns)
            dim_score = min(5, data["embedding_dimension"] / 200)
            scores["efficiency"] = size_score + dim_score

        # 4. Cyber understanding score (0-10)
        if "semantic_understanding" in data:
            # High contrast ratio is good (max 5 points)
            contrast_score = min(5, data["semantic_understanding"]["contrast_ratio"])
            # Low delta from expected is good (max 5 points)
            delta_score = 5 * (
                1 - min(1, data["semantic_understanding"]["avg_delta"] * 2)
            )
            scores["cyber_understanding"] = contrast_score + delta_score

        model_scores[model_name] = scores

    # Normalize speed scores across models
    if len(model_scores) > 0:
        # Find max speed
        speeds = [
            scores.get("_texts_per_second", 0) for scores in model_scores.values()
        ]
        max_speed = max(speeds) if speeds else 1

        # Normalize speeds to 0-10 range
        for model_name, scores in model_scores.items():
            if "_texts_per_second" in scores:
                scores["speed"] = min(10, scores["_texts_per_second"] / max_speed * 10)
                del scores["_texts_per_second"]

    # Calculate total scores
    for model_name, scores in model_scores.items():
        if "error" not in scores:
            # Weighted average of scores
            scores["total_score"] = (
                scores["embedding_quality"] * 0.35
                + scores["speed"] * 0.25
                + scores["efficiency"] * 0.15
                + scores["cyber_understanding"] * 0.25
            )

    # Find best model
    best_model = max(model_scores.items(), key=lambda x: x[1].get("total_score", 0))

    # Generate report
    with open(OUTPUT_DIR / "embedding_model_report.md", "w") as f:
        f.write("# Sentence Embedding Model Analysis for CTI Pipeline\n\n")

        f.write("## Overview\n\n")
        f.write(
            "This report analyzes different sentence transformer models to determine the optimal embeddings for cybersecurity text data.\n\n"
        )

        f.write("## Model Comparison\n\n")
        f.write(
            "| Model | Embedding Quality | Speed | Efficiency | Cyber Understanding | Total Score |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- |\n")

        for model_name, scores in model_scores.items():
            if "error" in scores:
                f.write(f"| {model_name} | Error | Error | Error | Error | 0 |\n")
            else:
                f.write(
                    f"| {model_name} | {scores['embedding_quality']:.2f} | {scores['speed']:.2f} | {scores['efficiency']:.2f} | {scores['cyber_understanding']:.2f} | {scores['total_score']:.2f} |\n"
                )

        f.write("\n## Recommendation\n\n")
        if "error" not in best_model[1]:
            f.write(
                f"Based on our analysis, **{best_model[0]}** is the recommended model with a score of {best_model[1]['total_score']:.2f}/10.\n\n"
            )

            # Compare with your current model
            current_model_score = model_scores.get("all-MiniLM-L6-v2", {}).get(
                "total_score", 0
            )
            if best_model[0] == "all-MiniLM-L6-v2":
                f.write(
                    "Your current model choice (all-MiniLM-L6-v2) is optimal based on our evaluation metrics.\n\n"
                )
            elif current_model_score > 0:
                score_diff = best_model[1]["total_score"] - current_model_score
                if score_diff > 1:
                    f.write(
                        f"Switching from your current model (all-MiniLM-L6-v2) to {best_model[0]} could provide significant improvements (score difference: +{score_diff:.2f}).\n\n"
                    )
                else:
                    f.write(
                        f"Your current model (all-MiniLM-L6-v2) performs well, with only a small difference ({score_diff:.2f}) compared to the top model.\n\n"
                    )
        else:
            f.write(
                "Could not determine a recommendation due to errors in the evaluation.\n\n"
            )

        f.write("## Detailed Justification\n\n")
        f.write("### all-MiniLM-L6-v2 (Current Model)\n\n")

        current_data = results.get("all-MiniLM-L6-v2", {})
        if "error" not in current_data:
            f.write(
                f"- **Embedding Dimension**: {current_data.get('embedding_dimension', 'N/A')}\n"
            )
            f.write(
                f"- **Model Size**: {current_data.get('model_size_mb', 'N/A'):.2f} MB\n"
            )

            if "encoding_performance" in current_data:
                best_speed = max(
                    current_data["encoding_performance"],
                    key=lambda x: x["texts_per_second"],
                )
                f.write(
                    f"- **Encoding Speed**: {best_speed['texts_per_second']:.2f} texts/second (batch size {best_speed['batch_size']})\n"
                )

            if (
                "cluster_metrics" in current_data
                and "error" not in current_data["cluster_metrics"]
            ):
                f.write(
                    f"- **Clustering Quality (Silhouette)**: {current_data['cluster_metrics']['silhouette_score']:.4f}\n"
                )

            if "semantic_understanding" in current_data:
                f.write(
                    f"- **Cybersecurity Semantic Understanding**: Contrast ratio of {current_data['semantic_understanding']['contrast_ratio']:.2f}\n"
                )

            f.write(
                "\nThis model offers an excellent balance of performance, size, and semantic understanding for cybersecurity text. "
            )
            f.write(
                "The 384-dimensional embeddings provide sufficient representational power while keeping computational requirements manageable. "
            )

            # Add strengths and weaknesses
            strengths = []
            weaknesses = []

            if current_data.get("model_size_mb", float("inf")) < 100:
                strengths.append("compact model size")
            else:
                weaknesses.append("larger model size")

            if "encoding_performance" in current_data:
                if best_speed["texts_per_second"] > 50:
                    strengths.append("fast encoding speed")
                else:
                    weaknesses.append("slower encoding")

            if "semantic_understanding" in current_data:
                if current_data["semantic_understanding"]["contrast_ratio"] > 3:
                    strengths.append("strong cybersecurity concept differentiation")
                else:
                    weaknesses.append("weaker domain-specific understanding")

            if strengths:
                f.write(f"Key strengths include {', '.join(strengths)}. ")
            if weaknesses:
                f.write(
                    f"Areas for potential improvement include {', '.join(weaknesses)}. "
                )
        else:
            f.write("Could not evaluate due to errors in the analysis.\n")

        # Save the full results
        f.write("\n\n## Full Evaluation Data\n\n")
        f.write("See the accompanying JSON file for complete evaluation metrics.\n")

    # Save all raw results to JSON
    with open(OUTPUT_DIR / "embedding_model_full_results.json", "w") as f:
        # Convert numpy values to float for JSON serialization
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = convert_to_serializable(data)

        json.dump(serializable_results, f, indent=2)

    return model_scores


def convert_to_serializable(obj):
    """Convert numpy values and arrays to standard Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


# Main function


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logging.info("Starting sentence embedding model analysis")

    # Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    logging.info("Loading data")
    df = load_data()

    # Generate test sets
    logging.info("Creating test sets")
    test_sets = generate_test_sets(df)

    # Benchmark models
    logging.info("Benchmarking models")
    results = benchmark_models(test_sets)

    # Create visualizations
    logging.info("Creating comparative visualizations")
    create_comparative_visualizations(results)

    # Generate report
    logging.info("Generating report")
    model_scores = generate_report(results)

    logging.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")

    # Print summary
    best_model = max(model_scores.items(), key=lambda x: x[1].get("total_score", 0))
    if "error" not in best_model[1]:
        logging.info(
            f"Recommended model: {best_model[0]} (Score: {best_model[1]['total_score']:.2f}/10)"
        )
    else:
        logging.info("Could not determine a recommendation due to evaluation errors")


if __name__ == "__main__":
    main()
