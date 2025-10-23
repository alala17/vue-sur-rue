#!/usr/bin/env python3
"""
Simple password-based authentication for admin panel
Bypasses Firebase quota issues for admin access
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt

# Admin credentials
ADMIN_EMAIL = "alexandre.airvault@gmail.com"
ADMIN_PASSWORD = "admin123"  # Change this to a secure password

# JWT secret key (in production, use a secure random key)
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-this')

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_admin_token(email: str) -> str:
    """Create JWT token for admin"""
    payload = {
        'email': email,
        'role': 'admin',
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_admin_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify admin JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def authenticate_admin(email: str, password: str) -> Optional[str]:
    """Authenticate admin user and return token"""
    if email == ADMIN_EMAIL and verify_password(password, hash_password(ADMIN_PASSWORD)):
        return create_admin_token(email)
    return None
