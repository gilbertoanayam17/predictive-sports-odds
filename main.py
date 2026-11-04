"""Demostración de extremo a extremo en la terminal (envoltorio de :mod:`predictive_odds.cli`).

Uso::

    python main.py                       # Premier League simulada (sin red ni API keys)
    python main.py --backtest            # backtest walk-forward
    python main.py --help                # todas las opciones

Tras ``pip install -e .`` también está disponible el comando ``predictive-odds``.
"""

from __future__ import annotations

import sys

from predictive_odds.cli import main

if __name__ == "__main__":
    sys.exit(main())
