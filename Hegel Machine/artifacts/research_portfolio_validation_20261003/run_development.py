"""Reuse the fixed portfolio observer/runner at this study output."""
import os
from pathlib import Path
import runpy
BASE = Path(__file__).resolve().parent
os.environ['HEGEL_PORTFOLIO_OUTPUT'] = str(BASE)
runpy.run_path(str(BASE.parent / 'research_portfolio_20261003/run_development.py'), run_name='__main__')
