"""Streamlit entrypoint for Door Detector.

Keep the launch command stable:

  streamlit run door_detector/review_app.py
"""

from __future__ import annotations

from door_detector.ui.app import main

# Streamlit executes this file as a script; run the app immediately.
main()
