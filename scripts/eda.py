"""Generate EDA + unsupervised-training visualizations for the README.

All paths are resolved against the project root so this script runs from
any working directory. Outputs are written to assets/.

Every cluster-based plot uses the human-readable segment names (Champions,
At Risk, Hibernating, ...) loaded from model_metadata.json, never raw
"Cluster 0/1/2" labels, so the README reader always knows what they are
looking at.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import torch

from src.constants import PROJECT_ROOT

ASSETS = PROJECT_ROOT / "assets"
ARTIFACTS = PROJECT_ROOT / "artifacts"

sns.set_theme(style="darkgrid", context="notebook")
mpl.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "axes.titlecolor": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "grid.color": "#1e293b",
    "font.family": "DejaVu Sans",
    "font.size": 11,
})
# Stable colors keyed by segment name so a segment is always the same color.
SEGMENT_COLORS = {
    "Champions": "#10b981",
    "Loyal Customers": "#3b82f6",
    "Potential Loyalists": "#06b6d4",
    "At Risk": "#f59e0b",
    "Hibernating": "#ef4444",
    "Lost": "#b91c1c",
    "Outliers": "#64748b",
}
FALLBACK = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981",
            "#06b6d4", "#ef4444", "#84cc16"]


def _color_for(name, idx=0):
    return SEGMENT_COLORS.get(name, FALLBACK[idx % len(FALLBACK)])


def _load_data():
    df = pd.read_csv(ARTIFACTS / "data_ingestion" / "customer_data.csv")
    feats = ["Recency", "Frequency", "Monetary"]
    df_log = df.copy()
    df_log[feats] = np.log1p(df[feats])
    scaler = StandardScaler()
    X = scaler.fit_transform(df_log[feats])
    return df, df_log, X, feats


def _load_model():
    return joblib.load(ARTIFACTS / "model_trainer" / "model.pkl")


def _load_segment_map():
    """Return {int_label: segment_name} from the trained model metadata."""
    meta_path = ARTIFACTS / "model_trainer" / "model_metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    active = meta.get("active_model", "kmeans")
    segments = meta.get("models", {}).get(active, {}).get("segments", {})
    return {int(k): v["name"] for k, v in segments.items()}


def _label_name(label, seg_map):
    name = seg_map.get(int(label), f"Cluster {label}")
    if int(label) == -1:
        return "Outliers"
    return name


def plot_distributions(df_log, df, feats):
    # Raw
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = ["#3b82f6", "#10b981", "#ec4899"]
    units = ["days", "purchases", "dollars"]
    for ax, col, c, u in zip(axes, feats, palette, units):
        sns.histplot(df[col], bins=50, kde=True, color=c, ax=ax, edgecolor=None)
        ax.set_title(f"{col} (raw)")
        ax.set_xlabel(u)
    fig.suptitle("RFM Distributions before log transform (heavily right-skewed)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "rfm_distributions_raw.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Log
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, c, u in zip(axes, feats, palette, units):
        sns.histplot(df_log[col], bins=50, kde=True, color=c, ax=ax, edgecolor=None)
        ax.set_title(f"{col} (after log1p)")
        ax.set_xlabel(f"log({col})")
    fig.suptitle("RFM Distributions after log transform (closer to normal)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(ASSETS / "rfm_distributions_log.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(df_log, feats):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(df_log[feats].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"color": "#0f172a", "weight": "bold", "size": 12})
    ax.set_title("Correlation between RFM features (log-space)")
    fig.tight_layout()
    fig.savefig(ASSETS / "rfm_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter_pca(X, labels, seg_map):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, k in enumerate(sorted(np.unique(labels))):
        m = labels == k
        name = _label_name(k, seg_map)
        ax.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.7,
                   color=_color_for(name, i), label=name, edgecolor="none")
    explained = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Principal component 1 (captures {explained[0]:.0f}% of the spread)")
    ax.set_ylabel(f"Principal component 2 (captures {explained[1]:.0f}% of the spread)")
    ax.set_title("Customer segments projected into 2D (PCA)")
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", title="Segment")
    fig.tight_layout()
    fig.savefig(ASSETS / "cluster_scatter_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter_tsne(X, labels, seg_map):
    sample = np.random.RandomState(42).choice(len(X), size=min(2000, len(X)), replace=False)
    Xs, ys = X[sample], labels[sample]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", max_iter=1000)
    Z = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, k in enumerate(sorted(np.unique(ys))):
        m = ys == k
        name = _label_name(k, seg_map)
        ax.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.7,
                   color=_color_for(name, i), label=name, edgecolor="none")
    ax.set_xlabel("t-SNE dimension 1 (no units, preserves local neighborhoods)")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Customer segments projected into 2D (t-SNE)")
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", title="Segment")
    fig.tight_layout()
    fig.savefig(ASSETS / "cluster_scatter_tsne.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_radar(X, labels, feats, seg_map):
    unique = sorted(np.unique(labels))
    centers = np.array([X[labels == k].mean(axis=0) for k in unique])
    # Normalize each feature to [0,1] across centroids for the radar
    mins, maxs = centers.min(axis=0), centers.max(axis=0)
    norm = (centers - mins) / (maxs - mins + 1e-9)
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (k, row) in enumerate(zip(unique, norm)):
        name = _label_name(k, seg_map)
        vals = row.tolist() + [row[0]]
        col = _color_for(name, i)
        ax.plot(angles, vals, color=col, linewidth=2.2, label=name)
        ax.fill(angles, vals, color=col, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feats, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title("Segment profiles (normalized centroids).\n"
                 "Each shape is a segment. Bigger = higher value on that feature.\n"
                 "Values are scaled 0 to 1 across segments for comparison.",
                 pad=25, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True,
              facecolor="#1e293b", edgecolor="#334155", title="Segment")
    fig.tight_layout()
    fig.savefig(ASSETS / "cluster_profiles_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_sizes(labels, seg_map):
    unique = sorted(np.unique(labels))
    names = [_label_name(k, seg_map) for k in unique]
    counts = [(labels == k).sum() for k in unique]
    colors = [_color_for(n, i) for i, n in enumerate(names)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="none")
    total = sum(counts)
    for b, c in zip(bars, counts):
        pct = c / total * 100
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.01,
                f"{c}\n({pct:.0f}%)", ha="center", color="#f8fafc",
                fontsize=10, weight="bold")
    ax.set_ylabel("Number of customers")
    ax.set_title("How many customers fall into each segment")
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    fig.savefig(ASSETS / "cluster_sizes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_elbow_and_silhouette(X):
    ks = list(range(2, 11))
    wcss, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ks, wcss, marker="o", color="#3b82f6", linewidth=2)
    axes[0].axvline(5, color="#ef4444", linestyle="--", alpha=0.7, label="Chosen K = 5")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Within-cluster sum of squares (lower = tighter clusters)")
    axes[0].set_title("Elbow method: where adding more clusters stops helping")
    axes[0].legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    axes[1].plot(ks, sil, marker="s", color="#8b5cf6", linewidth=2)
    axes[1].axvline(5, color="#ef4444", linestyle="--", alpha=0.7, label="Chosen K = 5")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score (higher = better separated)")
    axes[1].set_title("Silhouette analysis: how well-separated the clusters are")
    axes[1].legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
    fig.tight_layout()
    fig.savefig(ASSETS / "model_selection.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(X):
    """Re-run K-Means tracking inertia per iteration (matches the PyTorch impl)."""
    torch.manual_seed(42)
    Xt = torch.tensor(X, dtype=torch.float32)
    K = 5
    idx = torch.randperm(Xt.shape[0])[:K]
    centroids = Xt[idx].clone()
    inertias = []
    for _ in range(300):
        d = torch.cdist(Xt, centroids)
        labels = torch.argmin(d, dim=1)
        inertia = (d.gather(1, labels.unsqueeze(1)) ** 2).sum().item()
        inertias.append(inertia)
        new_c = torch.stack([Xt[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centroids[k]
                             for k in range(K)])
        if torch.allclose(centroids, new_c, rtol=1e-4):
            centroids = new_c
            break
        centroids = new_c
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(inertias) + 1), inertias, marker="o", color="#10b981", linewidth=2)
    ax.axvline(len(inertias), color="#ef4444", linestyle="--", alpha=0.6,
               label=f"Converged at iteration {len(inertias)}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Within-cluster sum of squares (lower = tighter clusters)")
    ax.set_title("K-Means convergence: the model stops when centroids stop moving")
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(X):
    """Silhouette + Davies-Bouldin for K-Means, DBSCAN, GMM side by side."""
    names, sils, dbs, kcounts = [], [], [], []

    # K-Means
    km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42).fit(X)
    names.append("K-Means")
    sils.append(silhouette_score(X, km.labels_))
    dbs.append(davies_bouldin_score(X, km.labels_))
    kcounts.append(len(np.unique(km.labels_)))

    # GMM
    gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42).fit(X)
    gl = gmm.predict(X)
    names.append("GMM")
    sils.append(silhouette_score(X, gl))
    dbs.append(davies_bouldin_score(X, gl))
    kcounts.append(len(np.unique(gl)))

    # DBSCAN (tuned: eps=0.5, min_samples=5)
    db = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1).fit(X)
    dl = db.labels_
    mask = dl != -1
    ncl = len(np.unique(dl[mask])) if mask.any() else 0
    names.append("DBSCAN")
    kcounts.append(ncl)
    if mask.sum() > 1 and ncl >= 2:
        sils.append(silhouette_score(X[mask], dl[mask]))
        dbs.append(davies_bouldin_score(X[mask], dl[mask]))
    else:
        sils.append(np.nan)
        dbs.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cols = ["#3b82f6", "#8b5cf6", "#ec4899"]
    xlabels = [f"{n}\n({k} clusters)" for n, k in zip(names, kcounts)]

    bars0 = axes[0].bar(xlabels, sils, color=cols, edgecolor="none")
    for i, v in enumerate(sils):
        if not np.isnan(v):
            axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", color="#f8fafc",
                         fontsize=11, weight="bold")
    axes[0].set_title("Silhouette score (higher = better separated clusters)")
    axes[0].set_ylabel("Silhouette score (range -1 to 1)")

    bars1 = axes[1].bar(xlabels, dbs, color=cols, edgecolor="none")
    for i, v in enumerate(dbs):
        if not np.isnan(v):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", color="#f8fafc",
                         fontsize=11, weight="bold")
    axes[1].set_title("Davies-Bouldin index (lower = better separated clusters)")
    axes[1].set_ylabel("Davies-Bouldin index")
    fig.tight_layout()
    fig.savefig(ASSETS / "model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    print("Loading data + model ...")
    df, df_log, X, feats = _load_data()
    model = _load_model()
    labels = model.predict(X)
    seg_map = _load_segment_map()
    print(f"Segment map: {seg_map}")

    print("Generating: distributions / correlation")
    plot_distributions(df_log, df, feats)
    plot_correlation(df_log, feats)

    print("Generating: cluster scatter (PCA / t-SNE)")
    plot_cluster_scatter_pca(X, labels, seg_map)
    plot_cluster_scatter_tsne(X, labels, seg_map)

    print("Generating: cluster profiles radar + sizes")
    plot_cluster_radar(X, labels, feats, seg_map)
    plot_cluster_sizes(labels, seg_map)

    print("Generating: model selection (elbow + silhouette)")
    plot_elbow_and_silhouette(X)

    print("Generating: model comparison (K-Means / GMM / DBSCAN)")
    plot_model_comparison(X)

    print("Generating: convergence curve")
    plot_convergence(X)

    print(f"Done. Assets written to {ASSETS}")


if __name__ == "__main__":
    main()
