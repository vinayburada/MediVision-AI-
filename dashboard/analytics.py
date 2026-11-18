"""
MediVision AI - Analytics Dashboard Engine
Interactive Plotly visualizations, confusion matrices, ROC curves, and performance benchmarking.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config.settings import settings
from database.db import db


class AnalyticsEngine:
    """Computes analytics aggregates and generates interactive Plotly visualizations."""

    def __init__(self):
        self.colors = {
            "NORMAL": settings.COLOR_SUCCESS,
            "PNEUMONIA": settings.COLOR_ALERT,
            "navy": settings.COLOR_PRIMARY,
            "teal": settings.COLOR_SECONDARY,
            "card_bg": settings.COLOR_CARD_BG,
            "text": settings.COLOR_TEXT_MAIN,
        }

    @staticmethod
    def get_summary_kpis() -> Dict[str, Any]:
        """Fetch high-level overview KPIs from SQLite."""
        stats = db.get_statistics()
        total = stats.get("total_analyses", 0)
        dist = stats.get("category_distribution", {})
        pneumonia_count = dist.get("PNEUMONIA", 0)
        normal_count = dist.get("NORMAL", 0)

        positive_rate = round((pneumonia_count / total * 100), 1) if total > 0 else 0.0

        return {
            "total_analyses": total,
            "normal_count": normal_count,
            "pneumonia_count": pneumonia_count,
            "positive_rate": f"{positive_rate}%",
            "avg_confidence": f"{stats.get('average_confidence', 0.0) * 100:.1f}%",
            "avg_latency_ms": f"{stats.get('average_latency_ms', 0.0):.1f} ms",
        }

    def plot_class_distribution(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """Donut chart showing class breakdown."""
        if df is None or df.empty:
            stats = db.get_statistics()
            counts = stats.get("category_distribution", {})
            labels = list(counts.keys())
            values = list(counts.values())
        else:
            counts = df["predicted_category"].value_counts()
            labels = counts.index.tolist()
            values = counts.values.tolist()

        if not labels or sum(values) == 0:
            fig = go.Figure()
            fig.update_layout(title="No classification data recorded yet")
            return fig

        color_map = [self.colors.get(lbl, "#64748B") for lbl in labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=color_map),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )])

        fig.update_layout(
            title="Prediction Class Distribution",
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", color=self.colors["text"]),
        )
        return fig

    def plot_confidence_distribution(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """Histogram showing model confidence spread."""
        if df is None or df.empty:
            analyses = db.list_analyses(limit=500)
            if not analyses:
                fig = go.Figure()
                fig.update_layout(title="No analyses available for confidence histogram")
                return fig
            df = pd.DataFrame(analyses)

        if "confidence_score" not in df.columns or df["confidence_score"].dropna().empty:
            fig = go.Figure()
            fig.update_layout(title="No confidence scores available")
            return fig

        fig = px.histogram(
            df,
            x="confidence_score",
            color="predicted_category",
            nbins=20,
            barmode="overlay",
            color_discrete_map={"NORMAL": self.colors["NORMAL"], "PNEUMONIA": self.colors["PNEUMONIA"]},
            title="Confidence Score Distribution by Class",
            labels={"confidence_score": "Confidence (0.0 to 1.0)", "count": "Frequency"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(family="sans-serif", color=self.colors["text"]),
        )
        return fig

    def plot_confusion_matrix(self, cm: List[List[int]], class_names: List[str] = settings.CLASSES) -> go.Figure:
        """Interactive confusion matrix heatmap."""
        z = np.array(cm)
        x = [f"Pred {c}" for c in class_names]
        y = [f"True {c}" for c in class_names]

        # Annotations text
        text = [[str(val) for val in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            colorscale=[[0, "#E2E8F0"], [1, self.colors["navy"]]],
            showscale=True,
        ))

        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(title="Predicted Class"),
            yaxis=dict(title="Actual Ground Truth", autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=40, l=60, r=20),
            font=dict(family="sans-serif", color=self.colors["text"]),
        )
        return fig

    def plot_model_comparison(self, densenet_metrics: Dict[str, Any], vit_metrics: Dict[str, Any]) -> go.Figure:
        """Bar chart comparing DenseNet121 and Vision Transformer metrics."""
        metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
        densenet_vals = [
            densenet_metrics.get("accuracy", 0.0),
            densenet_metrics.get("precision", 0.0),
            densenet_metrics.get("recall", 0.0),
            densenet_metrics.get("f1_score", 0.0),
        ]
        vit_vals = [
            vit_metrics.get("accuracy", 0.0),
            vit_metrics.get("precision", 0.0),
            vit_metrics.get("recall", 0.0),
            vit_metrics.get("f1_score", 0.0),
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="DenseNet121",
            x=metric_keys,
            y=densenet_vals,
            marker_color=self.colors["teal"],
            text=[f"{v:.3f}" for v in densenet_vals],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="Vision Transformer (ViT-B/16)",
            x=metric_keys,
            y=vit_vals,
            marker_color=self.colors["navy"],
            text=[f"{v:.3f}" for v in vit_vals],
            textposition="auto",
        ))

        fig.update_layout(
            title="Model Architecture Comparison (DenseNet121 vs ViT)",
            barmode="group",
            yaxis=dict(range=[0, 1.05], title="Score"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=40, r=20),
            font=dict(family="sans-serif", color=self.colors["text"]),
        )
        return fig

    def plot_timeline(self, df: Optional[pd.DataFrame] = None) -> go.Figure:
        """Volume of analyses processed over time."""
        if df is None or df.empty:
            records = db.list_analyses(limit=500)
            if not records:
                fig = go.Figure()
                fig.update_layout(title="No temporal analysis history")
                return fig
            df = pd.DataFrame(records)

        if "analysis_timestamp" not in df.columns or df.empty:
            fig = go.Figure()
            fig.update_layout(title="No timestamps recorded")
            return fig

        df_copy = df.copy()
        # Parse timestamp dates
        df_copy["date"] = pd.to_datetime(df_copy["analysis_timestamp"], errors="coerce").dt.date
        counts = df_copy.groupby(["date", "predicted_category"]).size().reset_index(name="count")

        fig = px.bar(
            counts,
            x="date",
            y="count",
            color="predicted_category",
            color_discrete_map={"NORMAL": self.colors["NORMAL"], "PNEUMONIA": self.colors["PNEUMONIA"]},
            title="Daily Clinical Evaluation Volume",
            labels={"date": "Analysis Date", "count": "Evaluations"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=40, r=20),
            font=dict(family="sans-serif", color=self.colors["text"]),
        )
        return fig


analytics_engine = AnalyticsEngine()
