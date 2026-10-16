#!/usr/bin/env python3
"""
Start the spotter web app and open it in the browser.

    python spotter_web.py              # http://127.0.0.1:8770
    python spotter_web.py --lan        # listen on all interfaces (phones / other computers)
    python spotter_web.py --lan --league --tracks fril-26-race-7,fril-27-first-track
                                       # the page to share with pilots: watch runs, upload + analyse their own
"""

import argparse
import os
import threading
import webbrowser

import uvicorn

from version import full_version


def main():
    ap = argparse.ArgumentParser(description="FPV spotter web app")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces, not only this computer")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--league", action="store_true",
                    help="page for pilots: race videos only, upload + analyse; no learning, removing or browsing")
    ap.add_argument("--tracks", default="", help="comma-separated track folders to serve (default: all)")
    args = ap.parse_args()
    os.environ["SPOTTER_LEAGUE"] = "1" if args.league else ""
    os.environ["SPOTTER_TRACKS"] = args.tracks
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"lazyspotter {full_version()} on {url}{' · league page' if args.league else ''}"
          f"{' · tracks ' + args.tracks if args.tracks else ''}  (Ctrl+C to stop)")
    uvicorn.run("web.server:app", host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
