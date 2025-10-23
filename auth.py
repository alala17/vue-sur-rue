#!/usr/bin/env python3
"""
Firebase Authentication and JWT Middleware
Handles token verification and user authentication
"""

import os
import json
import logging
from functools import wraps
from typing import Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, auth
import jwt
from flask import request, jsonify, g

logger = logging.getLogger("auth")

# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            return True
        
        # Get Firebase service account key from environment
        firebase_config = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
        if not firebase_config:
            logger.error("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set")
            return False
        
        # Parse the JSON config
        try:
            service_account_info = json.loads(firebase_config)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in FIREBASE_SERVICE_ACCOUNT_KEY")
            return False
        
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        
        logger.info("Firebase Admin SDK initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False

def verify_firebase_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify Firebase ID token and return decoded token"""
    try:
        if not firebase_admin._apps:
            logger.error("Firebase not initialized")
            return None
        
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        return decoded_token
        
    except auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase token")
        return None
    except auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase token")
        return None
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        return None

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Get user information from Firebase token"""
    decoded_token = verify_firebase_token(token)
    if not decoded_token:
        return None
    
    return {
        'uid': decoded_token.get('uid'),
        'email': decoded_token.get('email'),
        'email_verified': decoded_token.get('email_verified', False),
        'name': decoded_token.get('name'),
        'picture': decoded_token.get('picture')
    }

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        # Check if it's a Bearer token
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Verify token and get user info
        user_info = get_user_from_token(token)
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Store user info in Flask's g object for use in the route
        g.current_user = user_info
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_approved_user(f):
    """Decorator to require authenticated and approved user"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        token = auth_header[7:]
        user_info = get_user_from_token(token)
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Check if user is approved
        from user_manager import user_manager
        if not user_manager.is_user_approved(user_info['email']):
            return jsonify({'error': 'Access denied. Your account is not approved yet.'}), 403
        
        # Store user info
        g.current_user = user_info
        
        # Update last login
        user_manager.update_last_login(user_info['email'])
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_admin(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First check authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        token = auth_header[7:]
        user_info = get_user_from_token(token)
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Check if user is admin
        from user_manager import user_manager
        if not user_manager.is_user_admin(user_info['email']):
            return jsonify({'error': 'Admin privileges required'}), 403
        
        # Store user info
        g.current_user = user_info
        
        return f(*args, **kwargs)
    
    return decorated_function

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user from Flask's g object"""
    return getattr(g, 'current_user', None)
