"""
Update Python PATH to include project root directory.
Import this file before importing any other project module from a different directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
