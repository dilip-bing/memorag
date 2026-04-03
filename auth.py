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
            global_memory TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            last_login TEXT NOT NULL
        )
    """)
    # chats table — stores full chat JSON per user, one row per chat
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    # Migrate: add global_memory column if it doesn't exist yet
    try:
        conn.execute("ALTER TABLE users ADD COLUMN global_memory TEXT DEFAULT '[]'")
        conn.commit()
    except Exception:
        pass  # Column already exists
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


def get_global_memory(user_id: str) -> list:
    """Get a user's global memory cards."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT global_memory FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row or not row["global_memory"]:
        return []
    try:
        return json.loads(row["global_memory"])
    except Exception:
        return []


def update_global_memory(user_id: str, cards: list) -> bool:
    """Replace a user's global memory cards."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "UPDATE users SET global_memory = ? WHERE id = ?",
            (json.dumps(cards), user_id)
        )
        conn.commit()
        logger.info(f"Global memory updated for user {user_id}: {len(cards)} cards")
        return True
    except Exception as e:
        logger.error(f"Failed to update global memory: {e}")
        return False
    finally:
        conn.close()


# ── Chat persistence ───────────────────────────────────────────────────────

def get_chats(user_id: str) -> list:
    """Return all chats for a user, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT data FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        try:
            result.append(json.loads(row["data"]))
        except Exception:
            pass
    return result


def save_chat(user_id: str, chat: dict) -> bool:
    """Upsert a single chat (full JSON blob)."""
    chat_id = chat.get("id")
    if not chat_id:
        return False
    now = int(datetime.utcnow().timestamp() * 1000)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO chats (id, user_id, data, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   data = excluded.data,
                   updated_at = excluded.updated_at""",
            (chat_id, user_id, json.dumps(chat), now),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to save chat {chat_id}: {e}")
        return False
    finally:
        conn.close()


def delete_chat_db(user_id: str, chat_id: str) -> bool:
    """Delete a chat by ID (only if it belongs to user)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "DELETE FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to delete chat {chat_id}: {e}")
        return False
    finally:
        conn.close()


# Initialize database on module import
init_db()
