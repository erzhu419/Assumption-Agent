"""Use the already frozen companion runner for the fixed ten-root extension."""
import os
from pathlib import Path
import runpy
BASE=Path(__file__).resolve().parent
os.environ['HEGEL_PORTFOLIO_OUTPUT']=str(BASE)
runpy.run_path(BASE.parent/'portfolio_companion_20261004'/'run_development.py',run_name='__main__')
