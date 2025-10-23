#!/usr/bin/env python3
"""
Quick script to approve the first admin user
Run this once to approve yourself as the first admin
"""

import json
import os
from datetime import datetime

def approve_first_user(email):
    """Approve the first user and make them admin"""
    users_file = "users.json"
    
    if not os.path.exists(users_file):
        print(f"❌ {users_file} not found. Make sure the app has been deployed and at least one user has signed in.")
        return False
    
    try:
        # Load existing users
        with open(users_file, 'r') as f:
            users_data = json.load(f)
        
        if email not in users_data:
            print(f"❌ User {email} not found in database.")
            print("Available users:", list(users_data.keys()))
            return False
        
        # Update user status and role
        users_data[email]['status'] = 'approved'
        users_data[email]['role'] = 'admin'
        users_data[email]['updated_at'] = datetime.utcnow().isoformat()
        
        # Save updated users
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=2)
        
        print(f"✅ User {email} approved and promoted to admin!")
        print("You can now access the admin panel and manage other users.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔐 First User Approval Script")
    print("=" * 40)
    
    email = input("Enter your email address: ").strip()
    if not email:
        print("❌ Email is required")
        exit(1)
    
    if approve_first_user(email):
        print("\n🎉 Success! You can now:")
        print("1. Go to /admin on your deployed app")
        print("2. Sign in with your email")
        print("3. Manage other users")
    else:
        print("\n❌ Failed to approve user. Please check the error above.")
