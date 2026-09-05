"""Database utilities for storing detections and alerts."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATABASE_PATH


class DatabaseManager:
    """Simple SQLite-backed storage for detections and alerts."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DATABASE_PATH)
        self.logger = logging.getLogger("atm_detector.database")
        self.available = True
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            self.available = False
            self.logger.warning("Database storage is unavailable: %s", exc)

    def _initialize(self) -> None:
        """Create the required tables if they do not exist."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    image_path TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def log_detection(self, object_name: str, confidence: float, timestamp: Optional[str] = None) -> None:
        """Store a single detection event."""
        if not self.available:
            return
        stamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO detections (object_name, confidence, timestamp) VALUES (?, ?, ?)",
                    (object_name, confidence, stamp),
                )
                connection.commit()
        except sqlite3.Error as exc:
            self.logger.exception("Failed to store detection: %s", exc)

    def log_alert(self, threat_level: str, reason: str, image_path: str) -> None:
        """Store a generated alert event."""
        if not self.available:
            return
        now = datetime.now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO alerts (date, time, threat_level, reason, image_path) VALUES (?, ?, ?, ?, ?)",
                    (
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        threat_level,
                        reason,
                        image_path,
                    ),
                )
                connection.commit()
        except sqlite3.Error as exc:
            self.logger.exception("Failed to store alert: %s", exc)

    def get_alert_history(self) -> list[dict]:
        """Return the list of recent alerts."""
        if not self.available:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, date, time, threat_level, reason, image_path FROM alerts ORDER BY id DESC"
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            self.logger.warning("Failed to read alert history: %s", exc)
            return []
