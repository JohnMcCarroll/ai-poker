# ai-poker
A fun project exploring ML and RL techniques for the game of Texas Hold 'em

## Installation
- Create a venv using python 3.10.19 and pip 25.2. 
- For Windows Powershell:
- - `py -3.10 -m venv .venv`
- - `.venv\Scripts\Activate.ps1`
- - `pip install -e .` for `ai-poker`, `clubs`, and `clubs-gym` repositories in that order.
- - In the case of errors with `setup.py`, try: `pip install -e . --no-build-isolation --use-pep517`.


## Run MVP
Navigate to the repo root and run `python -m ai_poker.mvp.test_clubs_poker`.

