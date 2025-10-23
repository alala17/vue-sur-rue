#!/usr/bin/env python3
"""
User Management System for Vue sur Rue
Handles user authentication, approval, and role management
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("user_manager")

class UserStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REVOKED = "revoked"

class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"

@dataclass
class User:
    email: str
    firebase_uid: str
    status: UserStatus
    role: UserRole
    created_at: str
    updated_at: str
    invited_by: Optional[str] = None
    last_login: Optional[str] = None

class UserManager:
    def __init__(self, data_file: str = "users.json"):
        self.data_file = data_file
        self.users: Dict[str, User] = {}
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for email, user_data in data.items():
                        self.users[email] = User(
                            email=user_data['email'],
                            firebase_uid=user_data['firebase_uid'],
                            status=UserStatus(user_data['status']),
                            role=UserRole(user_data['role']),
                            created_at=user_data['created_at'],
                            updated_at=user_data['updated_at'],
                            invited_by=user_data.get('invited_by'),
                            last_login=user_data.get('last_login')
                        )
                logger.info(f"Loaded {len(self.users)} users from {self.data_file}")
            else:
                logger.info("No existing user data found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self.users = {}
    
    def save_users(self):
        """Save users to JSON file"""
        try:
            data = {}
            for email, user in self.users.items():
                data[email] = asdict(user)
                # Convert enums to strings for JSON serialization
                data[email]['status'] = user.status.value
                data[email]['role'] = user.role.value
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.users)} users to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def create_user(self, email: str, firebase_uid: str, invited_by: Optional[str] = None) -> User:
        """Create a new user"""
        now = datetime.utcnow().isoformat()
        user = User(
            email=email,
            firebase_uid=firebase_uid,
            status=UserStatus.PENDING,
            role=UserRole.USER,
            created_at=now,
            updated_at=now,
            invited_by=invited_by
        )
        self.users[email] = user
        self.save_users()
        logger.info(f"Created user: {email}")
        return user
    
    def get_user(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.users.get(email)
    
    def get_user_by_firebase_uid(self, firebase_uid: str) -> Optional[User]:
        """Get user by Firebase UID"""
        for user in self.users.values():
            if user.firebase_uid == firebase_uid:
                return user
        return None
    
    def update_user_status(self, email: str, status: UserStatus) -> bool:
        """Update user status"""
        if email in self.users:
            self.users[email].status = status
            self.users[email].updated_at = datetime.utcnow().isoformat()
            self.save_users()
            logger.info(f"Updated user {email} status to {status.value}")
            return True
        return False
    
    def update_user_role(self, email: str, role: UserRole) -> bool:
        """Update user role"""
        if email in self.users:
            self.users[email].role = role
            self.users[email].updated_at = datetime.utcnow().isoformat()
            self.save_users()
            logger.info(f"Updated user {email} role to {role.value}")
            return True
        return False
    
    def update_last_login(self, email: str) -> bool:
        """Update user's last login time"""
        if email in self.users:
            self.users[email].last_login = datetime.utcnow().isoformat()
            self.users[email].updated_at = datetime.utcnow().isoformat()
            self.save_users()
            return True
        return False
    
    def is_user_approved(self, email: str) -> bool:
        """Check if user is approved"""
        user = self.get_user(email)
        return user is not None and user.status == UserStatus.APPROVED
    
    def is_user_admin(self, email: str) -> bool:
        """Check if user is admin"""
        user = self.get_user(email)
        return user is not None and user.role == UserRole.ADMIN
    
    def get_all_users(self) -> List[User]:
        """Get all users"""
        return list(self.users.values())
    
    def get_pending_users(self) -> List[User]:
        """Get all pending users"""
        return [user for user in self.users.values() if user.status == UserStatus.PENDING]
    
    def delete_user(self, email: str) -> bool:
        """Delete a user"""
        if email in self.users:
            del self.users[email]
            self.save_users()
            logger.info(f"Deleted user: {email}")
            return True
        return False

# Global user manager instance
user_manager = UserManager()
