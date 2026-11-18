"""
MediVision AI - SQLite Database Management
Thread-safe connection handling, schema migration, and parameterized query execution.
"""
from pathlib import Path
import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from config.settings import settings


class DatabaseManager:
    """SQLite Database manager with parameterized queries and schema initialization."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row factory configured."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create database tables and performance indexes if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Analyses record table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    image_hash TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    normal_probability REAL NOT NULL,
                    pneumonia_probability REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    gradcam_image_path TEXT,
                    pdf_report_path TEXT,
                    ai_explanation_json TEXT,
                    notes TEXT
                );
            """)

            # Model Training runs audit table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    best_val_loss REAL,
                    val_accuracy REAL,
                    val_f1 REAL,
                    checkpoint_path TEXT,
                    created_at TEXT NOT NULL
                );
            """)

            # Create search and audit indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_id ON analyses(analysis_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(analysis_timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_category ON analyses(predicted_category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_hash ON analyses(image_hash);")

            conn.commit()

    def save_analysis(self, record: Dict[str, Any]) -> str:
        """
        Saves a new analysis record using safe parameterized SQL.
        Returns the analysis_id.
        """
        sql = """
            INSERT INTO analyses (
                analysis_id, image_hash, image_filename, image_path,
                predicted_category, confidence_score, normal_probability,
                pneumonia_probability, model_name, model_version,
                inference_time_ms, analysis_timestamp, gradcam_image_path,
                pdf_report_path, ai_explanation_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        analysis_id = record["analysis_id"]
        ai_json_str = (
            json.dumps(record.get("ai_explanation_json"))
            if isinstance(record.get("ai_explanation_json"), (dict, list))
            else record.get("ai_explanation_json", "")
        )

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (
                    analysis_id,
                    record.get("image_hash", ""),
                    record.get("image_filename", "unnamed.png"),
                    record.get("image_path", ""),
                    record.get("predicted_category", "UNKNOWN"),
                    float(record.get("confidence_score", 0.0)),
                    float(record.get("normal_probability", 0.0)),
                    float(record.get("pneumonia_probability", 0.0)),
                    record.get("model_name", settings.DEFAULT_MODEL_NAME),
                    record.get("model_version", settings.VERSION),
                    float(record.get("inference_time_ms", 0.0)),
                    record.get("analysis_timestamp", datetime.now(timezone.utc).isoformat()),
                    record.get("gradcam_image_path", ""),
                    record.get("pdf_report_path", ""),
                    ai_json_str,
                    record.get("notes", ""),
                ),
            )
            conn.commit()

        return analysis_id

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single analysis record by its unique ID."""
        sql = "SELECT * FROM analyses WHERE analysis_id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (analysis_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get("ai_explanation_json"):
                    try:
                        d["ai_explanation_parsed"] = json.loads(d["ai_explanation_json"])
                    except Exception:
                        d["ai_explanation_parsed"] = None
                return d
        return None

    def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
        search_query: str = "",
        category: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> List[Dict[str, Any]]:
        """
        List and filter analysis records with parameterized pagination and text search.
        """
        clauses = []
        params = []

        if search_query:
            clauses.append("(analysis_id LIKE ? OR image_filename LIKE ?)")
            wild = f"%{search_query.strip()}%"
            params.extend([wild, wild])

        if category and category.upper() in settings.CLASSES:
            clauses.append("predicted_category = ?")
            params.append(category.upper())

        if date_from:
            clauses.append("analysis_timestamp >= ?")
            params.append(date_from)

        if date_to:
            clauses.append("analysis_timestamp <= ?")
            params.append(date_to)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM analyses {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            for r in rows:
                item = dict(r)
                results.append(item)

        return results

    def delete_analysis(self, analysis_id: str, delete_files: bool = False) -> bool:
        """Deletes an analysis record and optionally associated artifacts."""
        record = self.get_analysis_by_id(analysis_id)
        if not record:
            return False

        if delete_files:
            for field in ["gradcam_image_path", "pdf_report_path"]:
                f_path = record.get(field)
                if f_path and Path(f_path).exists():
                    try:
                        Path(f_path).unlink()
                    except OSError:
                        pass

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
            conn.commit()

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate statistical metrics for dashboard analytics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total analyses count
            cursor.execute("SELECT COUNT(*) FROM analyses")
            total = cursor.fetchone()[0]

            # Category breakdown
            cursor.execute("""
                SELECT predicted_category, COUNT(*)
                FROM analyses
                GROUP BY predicted_category
            """)
            category_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Average confidence and latency
            cursor.execute("""
                SELECT AVG(confidence_score), AVG(inference_time_ms)
                FROM analyses
            """)
            avg_conf, avg_lat = cursor.fetchone()

            # Recent 10 analyses
            cursor.execute("""
                SELECT analysis_id, image_filename, predicted_category, confidence_score, analysis_timestamp
                FROM analyses
                ORDER BY id DESC LIMIT 10
            """)
            recent = [dict(r) for r in cursor.fetchall()]

        return {
            "total_analyses": total,
            "category_distribution": category_counts,
            "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "average_latency_ms": round(avg_lat, 2) if avg_lat else 0.0,
            "recent_analyses": recent,
        }


# Singleton database instance
db = DatabaseManager()
