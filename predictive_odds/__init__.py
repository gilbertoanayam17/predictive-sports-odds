"""predictive_odds: análisis estadístico deportivo, value bets y parlays.

Módulos principales:

* :mod:`predictive_odds.data_fetcher` — ingesta de datos y momios.
* :mod:`predictive_odds.statistical_models` — heurísticas, Poisson/Dixon-Coles y ML.
* :mod:`predictive_odds.parlay_builder` — valor, parlays y staking (Kelly).
* :mod:`predictive_odds.pipeline` — orquestación con API apta para UI.
"""

from .domain import Market, MatchPrediction, Outcome, Selection

__all__ = ["Market", "MatchPrediction", "Outcome", "Selection"]
__version__ = "0.2.0"
