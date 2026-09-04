"""
Unit and Integration Tests for ACL Graft Analysis
"""
import sys
from pathlib import Path

# Ensure Source is always in sys.path when running tests
SOURCE_DIR = Path(__file__).resolve().parent.parent / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
