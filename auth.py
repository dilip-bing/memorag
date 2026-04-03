"""
User authentication and database module
Handles Google OAuth token verification and user storage
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger("rag.auth")

# Database path
DB_PATH = Path(__file__).parent / "storage" / "users.db"


def init_db():
    """Initialize user database"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            picture TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"User database initialized at {DB_PATH}")


def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify Google JWT token and extract user info
    
    For now, we do basic JWT decoding without full verification
    In production, you should verify with Google's API
    """
    try:
        import base64
        
        # Split JWT
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        # Decode payload
        payload = parts[1]
        # Add padding if needed
        padding = len(payload) % 4
        if padding:
            payload += '=' * (4 - padding)
        
        decoded = base64.urlsafe_b64decode(payload)
        user_info = json.loads(decoded)
        
        # Validate required fields
        if not all(k in user_info for k in ['sub', 'email', 'name']):
            return None
        
        return {
            'id': user_info['sub'],
            'email': user_info['email'],
            'name': user_info['name'],
            'picture': user_info.get('picture', ''),
        }
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None


def get_or_create_user(user_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get existing user or create new one"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Check if user exists
    cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_info['id'],))
    row = cursor.fetchone()
    
    now = datetime.utcnow().isoformat()
    
    if row:
        # Update last login
        conn.execute(
            "UPDATE users SET last_login = ?, name = ?, picture = ? WHERE id = ?",
            (now, user_info['name'], user_info['picture'], user_info['id'])
        )
        conn.commit()
        user = dict(row)
        user['last_login'] = now
    else:
        # Create new user
        conn.execute(
            """INSERT INTO users (id, email, name, picture, created_at, last_login)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_info['id'], user_info['email'], user_info['name'], 
             user_info['picture'], now, now)
        )
        conn.commit()
        user = user_info.copy()
        user['created_at'] = now
        user['last_login'] = now
    
    conn.close()
    return user


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


# Initialize database on module import
init_db()
