NFL 2024 PPR & Snaps Dashboard

This Streamlit app visualizes weekly PPR fantasy points and snap counts for NFL players in the 2024 season using `nfl_data_py`.

Features
- Filters: position, team, player, opponent
- Dual-axis Plotly chart: PPR (bars) vs Snap Count (line)
- X-axis fixed to 18 weeks; BYE weeks shown with shaded background, values = 0 and opponent = BYE
- Opponent table for quick reference

Dev quickstart
1. Ensure Python 3.11 is available on your PATH.
2. Create a venv and install deps:
   - Windows PowerShell
     - `python -m venv .venv`
     - `. .venv/Scripts/Activate.ps1`
     - `pip install -r webapp/requirements.txt`
3. Run the app: `streamlit run webapp/app.py`

The app will print a local URL (e.g., http://localhost:8501) to access in your browser.

