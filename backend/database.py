import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pawprint.db')

def init_db():
    """Initialize the database with the scans table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            animal TEXT NOT NULL,
            confidence REAL NOT NULL,
            age_group TEXT,
            wildness TEXT,
            risk TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_scan(data):
    """Save a scan result to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO scans (timestamp, animal, confidence, age_group, wildness, risk)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        data.get('animal'),
        data.get('confidence'),
        data.get('estimated_age_group'),
        data.get('wildness_level'),
        data.get('human_risk_level')
    ))
    conn.commit()
    conn.close()

def get_recent_scans(limit=10):
    """Retrieve recent scans from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
