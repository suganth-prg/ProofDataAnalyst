"""
Runner script for Proof-Carrying Data Analyst UI.
Usage:
    python run_ui.py
or:
    streamlit run app.py
"""

import sys
import subprocess

def main():
    print("=" * 60)
    print("Starting Proof-Carrying Data Analyst UI...")
    print("URL will open in your browser (usually http://localhost:8501)")
    print("=" * 60)
    
    # Try importing and running via streamlit.web.cli
    try:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", "app.py", "--server.headless=false"]
        sys.exit(stcli.main())
    except ImportError:
        # Fallback to subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
