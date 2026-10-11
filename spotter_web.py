#!/usr/bin/env python3
"""
Start the spotter web app and open it in the browser.

    python spotter_web.py              # http://127.0.0.1:8770
    python spotter_web.py --lan        # listen on all interfaces (phones / other computers)
"""

import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="FPV spotter web app")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces, not only this computer")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Spotter web app on {url}  (Ctrl+C to stop)")
    uvicorn.run("web.server:app", host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
