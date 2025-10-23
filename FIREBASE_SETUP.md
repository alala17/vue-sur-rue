# Firebase Auth Setup Guide

## 🔥 Firebase Project Setup

### 1. Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Enter project name: `vue-sur-rue` (or your preferred name)
4. Enable Google Analytics (optional)
5. Click "Create project"

### 2. Enable Authentication
1. In Firebase Console, go to "Authentication"
2. Click "Get started"
3. Go to "Sign-in method" tab
4. Enable "Email/Password" provider
5. **Important**: Enable "Email link (passwordless sign-in)" option
6. Save changes

### 3. Get Firebase Configuration
1. Go to Project Settings (gear icon)
2. Scroll down to "Your apps"
3. Click "Add app" → Web app (</> icon)
4. Register app with name: `vue-sur-rue-web`
5. Copy the Firebase configuration object

### 4. Update Frontend Configuration
Replace the placeholder config in both `frontend.html` and `admin.html`:

```javascript
const firebaseConfig = {
    apiKey: "your-actual-api-key",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "your-actual-app-id"
};
```

### 5. Generate Service Account Key
1. Go to Project Settings → Service accounts
2. Click "Generate new private key"
3. Download the JSON file
4. Copy the entire JSON content

### 6. Set Environment Variables in Railway
Add these environment variables in your Railway project:

```
FIREBASE_SERVICE_ACCOUNT_KEY={"type":"service_account","project_id":"your-project",...}
```

**Important**: The entire JSON must be on one line as a string.

## 🚀 Deployment Steps

### 1. Update Configuration Files
- Replace Firebase config in `frontend.html` (line ~229)
- Replace Firebase config in `admin.html` (line ~15)

### 2. Set Railway Environment Variables
- `FIREBASE_SERVICE_ACCOUNT_KEY`: Your service account JSON (as string)
- `PINECONE_API_KEY`: Your existing Pinecone key
- `PINECONE_INDEX_NAME`: Your existing Pinecone index

### 3. Deploy to Railway
```bash
git add .
git commit -m "Add Firebase Auth system with user management"
git push origin main
```

## 👤 Admin Setup

### 1. First Admin User
1. Deploy the application
2. Go to `/admin` on your deployed site
3. Sign in with your email
4. The system will create your user account with "pending" status
5. You'll need to manually approve yourself in the database

### 2. Manual Admin Approval (First Time)
Since you'll be the first user, you'll need to manually approve yourself:

1. SSH into your Railway deployment or access the database
2. Edit the `users.json` file
3. Change your user's status from "pending" to "approved"
4. Change your role from "user" to "admin"

### 3. Alternative: Database Access
If you have database access, you can run:
```python
from user_manager import user_manager, UserStatus, UserRole
user_manager.update_user_status("your-email@example.com", UserStatus.APPROVED)
user_manager.update_user_role("your-email@example.com", UserRole.ADMIN)
```

## 🔐 How It Works

### User Flow
1. **Sign In**: User enters email → receives magic link → clicks link → signed in
2. **Verification**: Backend verifies Firebase token and checks user status
3. **Access Control**: Only approved users can use the search functionality
4. **Admin Panel**: Admins can manage user access at `/admin`

### Security Features
- ✅ Firebase JWT token verification
- ✅ User approval system (pending/approved/revoked)
- ✅ Role-based access (user/admin)
- ✅ Magic link authentication (no passwords)
- ✅ Automatic user creation on first sign-in
- ✅ Admin panel for user management

### API Endpoints
- `POST /api/auth/verify` - Verify user authentication
- `GET /api/admin/users` - Get all users (admin only)
- `PUT /api/admin/users/{email}/status` - Update user status (admin only)
- `PUT /api/admin/users/{email}/role` - Update user role (admin only)
- `DELETE /api/admin/users/{email}` - Delete user (admin only)

## 🛠️ Troubleshooting

### Common Issues
1. **"Firebase not initialized"** - Check `FIREBASE_SERVICE_ACCOUNT_KEY` environment variable
2. **"Invalid token"** - Ensure Firebase config is correct in frontend
3. **"Access denied"** - User needs to be approved by admin
4. **"Admin privileges required"** - User needs admin role

### Testing Locally
1. Set environment variables in your local environment
2. Run `python3 backend.py`
3. Open `http://localhost:8080`
4. Test authentication flow

## 📝 Notes
- Users are automatically created on first sign-in
- All users start with "pending" status
- Only approved users can access the search functionality
- Admins can manage users through the `/admin` panel
- The system uses JSON file storage (can be upgraded to database later)
