"""
Vercel Serverless Entrypoint for Project RM
Routes serverless HTTP requests to DashboardHandler in server.py
"""

import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import DashboardHandler


class handler(DashboardHandler):
    """Vercel serverless request handler."""
    pass
