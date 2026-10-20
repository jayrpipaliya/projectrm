"""
Launcher for Project RM Web Dashboard.
Opens the local quant visualization in the default web browser.
"""

import webbrowser
import threading
import time
import config
from server import run_server

def open_browser():
    time.sleep(1.2)
    url = f"http://localhost:{config.PORT}"
    print(f"[*] Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    run_server()
