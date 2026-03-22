"""
Simple password-based authentication for the CHHAT Streamlit app.
Reads credentials from environment variables APP_USERNAME and APP_PASSWORD.
"""

import os
import streamlit as st


def check_auth() -> bool:
    """Show login form if not authenticated. Returns True if authenticated."""

    # Logout button in sidebar (only when logged in)
    if st.session_state.get("authenticated"):
        with st.sidebar:
            if st.button("Logout", key="logout_btn"):
                st.session_state["authenticated"] = False
                st.session_state["auth_username"] = None
                st.rerun()
        return True

    # Credentials from environment
    valid_username = os.getenv("APP_USERNAME", "admin")
    valid_password = os.getenv("APP_PASSWORD", "chhat2026")

    # Centered login form
    st.markdown(
        """
        <style>
            .login-container {
                max-width: 400px;
                margin: 4rem auto;
                padding: 2rem;
                background: #f8f9fa;
                border-radius: 12px;
                border: 1px solid #dee2e6;
            }
            .login-title {
                font-size: 1.6rem;
                font-weight: 700;
                color: #1a1a2e;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            .login-subtitle {
                font-size: 0.95rem;
                color: #6c757d;
                text-align: center;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-title">CHHAT Brand Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Please sign in to continue</div>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                if username == valid_username and password == valid_password:
                    st.session_state["authenticated"] = True
                    st.session_state["auth_username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    return False
