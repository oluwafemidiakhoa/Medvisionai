
# -*- coding: utf-8 -*-
"""
auth.py - Simple authentication for RadVision AI
================================================

Provides basic authentication capabilities to protect the application.
Uses secure hashing for password storage.
"""

import hashlib
import hmac
import os
import json
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Constants
AUTH_CONFIG_FILE = "auth_config.json"
SESSION_DURATION = timedelta(hours=8)
DEFAULT_USERNAME = "admin"
AUTH_COOKIE_NAME = "radvision_auth"

# Initialize authentication configuration
def init_auth_config() -> None:
    """Initialize the authentication configuration if it doesn't exist."""
    if not os.path.exists(AUTH_CONFIG_FILE):
        # Generate a secure random salt
        salt = secrets.token_hex(16)
        
        # Create default admin account
        default_password = "radvision"  # Users should change this immediately
        password_hash = hash_password(default_password, salt)
        
        config = {
            "salt": salt,
            "users": {
                DEFAULT_USERNAME: {
                    "password_hash": password_hash,
                    "is_admin": True,
                    "created_at": datetime.now().isoformat()
                }
            }
        }
        
        # Save configuration
        with open(AUTH_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info("Created default authentication configuration")

def hash_password(password: str, salt: str) -> str:
    """
    Securely hash a password with the given salt.
    
    Args:
        password: The plaintext password
        salt: The salt string
        
    Returns:
        Hexadecimal string of the password hash
    """
    # Convert inputs to bytes
    password_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    
    # Use HMAC-SHA256 for secure password hashing
    return hmac.new(salt_bytes, password_bytes, hashlib.sha256).hexdigest()

def verify_password(username: str, password: str) -> bool:
    """
    Verify a password for a given username.
    
    Args:
        username: The username to check
        password: The plaintext password to verify
        
    Returns:
        True if password is correct, False otherwise
    """
    if not os.path.exists(AUTH_CONFIG_FILE):
        init_auth_config()
        
    try:
        # Load configuration
        with open(AUTH_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        # Check if user exists
        if username not in config["users"]:
            logger.warning(f"Login attempt for unknown user: {username}")
            return False
            
        # Get user data
        user_data = config["users"][username]
        
        # Get salt
        salt = config["salt"]
        
        # Hash the provided password
        password_hash = hash_password(password, salt)
        
        # Compare with stored hash
        if hmac.compare_digest(password_hash, user_data["password_hash"]):
            logger.info(f"Successful login for user: {username}")
            return True
        else:
            logger.warning(f"Failed login attempt for user: {username}")
            return False
    except Exception as e:
        logger.error(f"Error during password verification: {e}")
        return False

def is_authenticated() -> Tuple[bool, Optional[str]]:
    """
    Check if the current session is authenticated.
    
    Returns:
        Tuple of (is_authenticated, username)
    """
    # Check for authentication in session state
    auth_data = st.session_state.get("auth_data")
    if auth_data:
        # Verify expiration
        expiry = datetime.fromisoformat(auth_data["expiry"])
        if datetime.now() < expiry:
            return True, auth_data["username"]
    
    return False, None

def authenticate(username: str, password: str) -> bool:
    """
    Authenticate a user and set up the session.
    
    Args:
        username: The username to authenticate
        password: The plaintext password
        
    Returns:
        True if authentication successful, False otherwise
    """
    if verify_password(username, password):
        # Set expiration time
        expiry = datetime.now() + SESSION_DURATION
        
        # Create authentication data
        auth_data = {
            "username": username,
            "expiry": expiry.isoformat()
        }
        
        # Store in session state
        st.session_state.auth_data = auth_data
        
        return True
    return False

def logout() -> None:
    """Log out the current user."""
    # Clear session state
    if "auth_data" in st.session_state:
        del st.session_state.auth_data

def render_login_page() -> None:
    """Render the login page UI."""
    st.title("🔒 RadVision AI Login")
    
    st.markdown("""
    Welcome to RadVision AI. Please log in to continue.
    
    If this is your first time, use the default credentials:
    - Username: admin
    - Password: radvision
    
    Please change your password after first login.
    """)
    
    # Check if auth config exists
    if not os.path.exists(AUTH_CONFIG_FILE):
        init_auth_config()
        st.info("First-time setup complete. Default admin account created.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate(username, password):
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password")

def render_user_management() -> None:
    """Render the user management UI for admins."""
    is_auth, username = is_authenticated()
    if not is_auth:
        return
        
    # Check if user is admin
    try:
        with open(AUTH_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        is_admin = config["users"].get(username, {}).get("is_admin", False)
        if not is_admin:
            st.warning("You don't have permission to manage users")
            return
    except Exception as e:
        st.error(f"Error checking permissions: {e}")
        return
        
    st.subheader("👥 User Management")
    
    # Current users
    with st.expander("Current Users", expanded=True):
        try:
            with open(AUTH_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
            users = config["users"]
            
            for user, data in users.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(f"Username: {user}")
                with col2:
                    st.text(f"Admin: {'✅' if data.get('is_admin') else '❌'}")
                with col3:
                    if user != username:  # Don't allow deleting self
                        if st.button("Delete", key=f"del_{user}"):
                            del users[user]
                            with open(AUTH_CONFIG_FILE, 'w') as f:
                                json.dump(config, f, indent=2)
                            st.success(f"User {user} deleted")
                            st.rerun()
        except Exception as e:
            st.error(f"Error loading users: {e}")
    
    # Add new user
    with st.expander("Add New User"):
        with st.form("add_user_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("Password", type="password")
            is_admin = st.checkbox("Is Admin")
            submit = st.form_submit_button("Add User")
            
            if submit:
                try:
                    if not new_username or not new_password:
                        st.error("Username and password are required")
                    else:
                        with open(AUTH_CONFIG_FILE, 'r') as f:
                            config = json.load(f)
                            
                        if new_username in config["users"]:
                            st.error(f"User {new_username} already exists")
                        else:
                            # Add new user
                            password_hash = hash_password(new_password, config["salt"])
                            config["users"][new_username] = {
                                "password_hash": password_hash,
                                "is_admin": is_admin,
                                "created_at": datetime.now().isoformat()
                            }
                            
                            with open(AUTH_CONFIG_FILE, 'w') as f:
                                json.dump(config, f, indent=2)
                                
                            st.success(f"User {new_username} added successfully")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error adding user: {e}")
    
    # Change password
    with st.expander("Change Your Password"):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submit = st.form_submit_button("Change Password")
            
            if submit:
                try:
                    if not current_password or not new_password or not confirm_password:
                        st.error("All fields are required")
                    elif new_password != confirm_password:
                        st.error("New passwords don't match")
                    elif not verify_password(username, current_password):
                        st.error("Current password is incorrect")
                    else:
                        with open(AUTH_CONFIG_FILE, 'r') as f:
                            config = json.load(f)
                            
                        # Update password
                        password_hash = hash_password(new_password, config["salt"])
                        config["users"][username]["password_hash"] = password_hash
                        
                        with open(AUTH_CONFIG_FILE, 'w') as f:
                            json.dump(config, f, indent=2)
                            
                        st.success("Password changed successfully")
                except Exception as e:
                    st.error(f"Error changing password: {e}")
