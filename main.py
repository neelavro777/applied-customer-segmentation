import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.constants import PROJECT_ROOT
from src.pipeline.prediction_pipeline import PredictionPipeline

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_trainer" / "model_metadata.json"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _load_meta():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {"active_model": "kmeans", "models": {}, "metrics": {}}


META = _load_meta()


# Plain-English presets matching the trained centroids (original RFM scale).
# Labels are deliberately non-technical so business users understand them.
PRESETS = [
    {"label": "Best customer", "recency": 8, "frequency": 250, "monetary": 5000},
    {"label": "Loyal regular", "recency": 12, "frequency": 75, "monetary": 1100},
    {"label": "Slipping fan", "recency": 100, "frequency": 55, "monetary": 1000},
    {"label": "New buyer", "recency": 30, "frequency": 16, "monetary": 290},
    {"label": "Gone quiet", "recency": 205, "frequency": 10, "monetary": 207},
]


def _model_display(name: str) -> str:
    return {"kmeans": "K-Means", "dbscan": "DBSCAN", "gmm": "Gaussian Mixture"}.get(name, name)


def _interpret_delta(feature: str, inp: float, centroid: float) -> str:
    """Plain-language comparison of the customer vs the typical customer in their segment."""
    if centroid == 0:
        return f"{feature}: {inp:.0f}"
    pct = (inp - centroid) / abs(centroid) * 100
    if feature == "Recency":
        # For recency, lower is better.
        if abs(pct) < 10:
            qual = "about the same as"
        elif pct < 0:
            qual = f"better than ({abs(pct):.0f}% more recent than)"
        else:
            qual = f"worse than ({pct:.0f}% less recent than)"
        return f"Days since last purchase: {inp:.0f} - {qual} the typical {feature.lower()} of this group ({centroid:.0f})."
    else:
        unit = "purchases" if feature == "Frequency" else "$ spent"
        if abs(pct) < 10:
            qual = "about the same as"
        elif pct > 0:
            qual = f"higher than ({pct:.0f}% more than)"
        else:
            qual = f"lower than ({abs(pct):.0f}% less than)"
        return f"{feature}: {inp:.0f} {unit} - {qual} the group average ({centroid:.0f})."


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    active = META.get("active_model", "kmeans")
    model_info = META.get("models", {}).get(active, {})
    metrics = META.get("metrics", {}).get(active, {})
    total = sum(s.get("size", 0) for s in model_info.get("segments", {}).values())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "presets": PRESETS,
            "model_display": _model_display(active),
            "n_clusters": model_info.get("n_clusters", 0),
            "n_customers": total,
            "silhouette": metrics.get("silhouette_score"),
            "davies_bouldin": metrics.get("davies_bouldin_index"),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_segment(request: Request,
                          recency: float = Form(...),
                          frequency: float = Form(...),
                          monetary: float = Form(...)):
    base_ctx = {
        "request": request,
        "presets": PRESETS,
        "model_display": _model_display(META.get("active_model", "kmeans")),
        "n_clusters": META.get("models", {}).get(META.get("active_model", "kmeans"), {}).get("n_clusters", 0),
        "n_customers": sum(s.get("size", 0) for s in META.get("models", {}).get(META.get("active_model", "kmeans"), {}).get("segments", {}).values()),
        "silhouette": META.get("metrics", {}).get(META.get("active_model", "kmeans"), {}).get("silhouette_score"),
        "davies_bouldin": META.get("metrics", {}).get(META.get("active_model", "kmeans"), {}).get("davies_bouldin_index"),
    }
    try:
        pipeline = PredictionPipeline()
        cluster_id, distances, seg, model_name = pipeline.predict_with_distances(recency, frequency, monetary)

        centroid_rfm = seg.get("centroid_rfm", [0, 0, 0])
        # Plain-language comparison vs the typical customer in this segment.
        comparisons = [
            _interpret_delta("Recency", recency, centroid_rfm[0]),
            _interpret_delta("Frequency", frequency, centroid_rfm[1]),
            _interpret_delta("Monetary", monetary, centroid_rfm[2]),
        ]

        # Normalized input vs centroid for the radar chart (0-1 across all centroids + input).
        active = META.get("active_model", "kmeans")
        all_centroids = [v.get("centroid_rfm", [0, 0, 0])
                         for v in META.get("models", {}).get(active, {}).get("segments", {}).values()]
        if all_centroids:
            arr = np.array(all_centroids + [[recency, frequency, monetary]])
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            norm_input = ((np.array([recency, frequency, monetary]) - lo) / (hi - lo + 1e-9)).tolist()
            norm_centroid = ((np.array(centroid_rfm) - lo) / (hi - lo + 1e-9)).tolist()
        else:
            norm_input = norm_centroid = [0, 0, 0]

        # Confidence: softmax-negative of distances.
        d_vals = np.array(list(distances.values()), dtype=float)
        conf = float(np.exp(-d_vals.min()) / np.exp(-d_vals).sum())

        # Per-cluster distance bars (for the chart).
        seg_map = {k: v.get("name", f"Cluster {k}")
                   for k, v in META.get("models", {}).get(active, {}).get("segments", {}).items()}
        per_cluster = []
        for label, dist in distances.items():
            per_cluster.append({
                "label": label,
                "name": seg_map.get(label, f"Cluster {label}"),
                "distance": float(dist),
                "is_assigned": str(cluster_id) == str(label),
            })
        per_cluster.sort(key=lambda c: c["distance"])

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                **base_ctx,
                "cluster_id": int(cluster_id),
                "segment_name": seg.get("name", "Unknown"),
                "who": seg.get("who", ""),
                "churn_risk": seg.get("churn_risk", "Unknown"),
                "customer_value": seg.get("customer_value", "Unknown"),
                "actions": seg.get("actions", []),
                "tactic": seg.get("tactic", ""),
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
                "confidence": round(conf * 100, 1),
                "comparisons": comparisons,
                "centroid_rfm": centroid_rfm,
                "norm_input": norm_input,
                "norm_centroid": norm_centroid,
                "per_cluster": per_cluster,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={**base_ctx, "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
