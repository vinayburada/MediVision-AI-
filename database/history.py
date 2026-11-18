"""
MediVision AI - History Management Service
Data querying, filtering, table formatting, and audit management.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from database.db import db, DatabaseManager


class HistoryManager:
    """Service layer interfacing UI and API with database storage."""

    def __init__(self, db_instance: DatabaseManager = db):
        self.db = db_instance

    def get_history_dataframe(
        self,
        limit: int = 100,
        search: str = "",
        category: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> pd.DataFrame:
        """Fetch analyses as a clean pandas DataFrame ready for Streamlit display."""
        records = self.db.list_analyses(
            limit=limit,
            search_query=search,
            category=category,
            date_from=date_from,
            date_to=date_to,
        )
        if not records:
            return pd.DataFrame(columns=[
                "analysis_id", "image_filename", "predicted_category",
                "confidence_score", "normal_probability", "pneumonia_probability",
                "inference_time_ms", "analysis_timestamp"
            ])

        df = pd.DataFrame(records)
        display_cols = [
            "analysis_id", "image_filename", "predicted_category",
            "confidence_score", "normal_probability", "pneumonia_probability",
            "inference_time_ms", "analysis_timestamp"
        ]
        # Keep available cols
        avail_cols = [c for c in display_cols if c in df.columns]
        return df[avail_cols]

    def get_record(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full details for an analysis."""
        return self.db.get_analysis_by_id(analysis_id)

    def delete_record(self, analysis_id: str, delete_files: bool = True) -> bool:
        """Delete an analysis record."""
        return self.db.delete_analysis(analysis_id, delete_files=delete_files)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Fetch key performance indicators for the overview dashboard."""
        return self.db.get_statistics()


history_manager = HistoryManager()
