"""
Simple password-based authentication for the CHHAT Streamlit app.
"""

import os
import streamlit as st


def check_auth() -> bool:
    """Show login form if not authenticated. Returns True if authenticated."""

    if st.session_state.get("authenticated"):
        with st.sidebar:
            if st.button("Logout", key=f"logout_{id(check_auth)}"):
                st.session_state["authenticated"] = False
                st.session_state["auth_username"] = None
                st.rerun()
        return True

    # Not authenticated — hide sidebar and show login
    # Use CSS to hide the sidebar when not logged in
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none; }
            [data-testid="stSidebarNav"] { display: none; }
            header[data-testid="stHeader"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    valid_username = os.getenv("APP_USERNAME", "admin")
    valid_password = os.getenv("APP_PASSWORD", "chhat2026")

    # Centered login form
    st.markdown("")
    st.markdown("")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image("chhat-logo.png", width=160)
        st.markdown("#### Please sign in to continue")
        st.markdown("")

        with st.form("login_form", clear_on_submit=False):
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
