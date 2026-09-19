"""
EPICS ROP Tri-Modal Retinal Diagnostic Web Application Server.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.web import app

if __name__ == "__main__":
    print("Starting EPICS ROP Tri-Modal Diagnostic Platform on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
