"""Emparejamiento de nombres entre football-data.co.uk y The Odds API.

Los nombres de cada fuente son los esperados según sus convenciones; no se
pudieron verificar en vivo (sin red). Si alguno no casa en la práctica, la app
lo muestra como "Sin historial" y se corrige con un alias.
"""

from __future__ import annotations

import pytest

from predictive_odds.data_fetcher import TeamNameNormalizer

# (nombre en football-data.co.uk, nombre en The Odds API)
PAIRS: dict[str, list[tuple[str, str]]] = {
    "LA_LIGA": [
        ("Alaves", "Alavés"), ("Ath Bilbao", "Athletic Bilbao"), ("Ath Madrid", "Atlético Madrid"),
        ("Barcelona", "Barcelona"), ("Celta", "Celta Vigo"), ("Elche", "Elche CF"), ("Espanol", "Espanyol"),
        ("Getafe", "Getafe"), ("Girona", "Girona"), ("Levante", "Levante"), ("Mallorca", "Mallorca"),
        ("Osasuna", "CA Osasuna"), ("Vallecano", "Rayo Vallecano"), ("Betis", "Real Betis"),
        ("Real Madrid", "Real Madrid"), ("Oviedo", "Real Oviedo"), ("Sociedad", "Real Sociedad"),
        ("Sevilla", "Sevilla"), ("Valencia", "Valencia"), ("Villarreal", "Villarreal"),
    ],
    "SERIE_A": [
        ("Atalanta", "Atalanta BC"), ("Bologna", "Bologna"), ("Cagliari", "Cagliari"), ("Como", "Como"),
        ("Cremonese", "Cremonese"), ("Fiorentina", "Fiorentina"), ("Genoa", "Genoa"),
        ("Verona", "Hellas Verona FC"), ("Inter", "Inter Milan"), ("Juventus", "Juventus"), ("Lazio", "Lazio"),
        ("Lecce", "Lecce"), ("Milan", "AC Milan"), ("Napoli", "Napoli"), ("Parma", "Parma"), ("Pisa", "Pisa"),
        ("Roma", "AS Roma"), ("Sassuolo", "Sassuolo"), ("Torino", "Torino"), ("Udinese", "Udinese"),
    ],
    "BUNDESLIGA": [
        ("Augsburg", "FC Augsburg"), ("Leverkusen", "Bayer Leverkusen"), ("Bayern Munich", "Bayern Munich"),
        ("Dortmund", "Borussia Dortmund"), ("M'gladbach", "Borussia Monchengladbach"),
        ("Ein Frankfurt", "Eintracht Frankfurt"), ("FC Koln", "1. FC Köln"), ("Freiburg", "SC Freiburg"),
        ("Hamburg", "Hamburger SV"), ("Heidenheim", "1. FC Heidenheim"), ("Hoffenheim", "TSG Hoffenheim"),
        ("Mainz", "FSV Mainz 05"), ("RB Leipzig", "RB Leipzig"), ("St Pauli", "FC St. Pauli"),
        ("Union Berlin", "Union Berlin"), ("Stuttgart", "VfB Stuttgart"), ("Werder Bremen", "Werder Bremen"),
        ("Wolfsburg", "VfL Wolfsburg"),
    ],
    "LIGUE_1": [
        ("Paris SG", "Paris Saint Germain"), ("Paris FC", "Paris FC"), ("Marseille", "Marseille"),
        ("Lyon", "Lyon"), ("Monaco", "AS Monaco"), ("Lens", "RC Lens"), ("Lille", "Lille"), ("Rennes", "Rennes"),
        ("Le Havre", "Le Havre"), ("Strasbourg", "Strasbourg"),
    ],
    "EREDIVISIE": [
        ("Ajax", "Ajax"), ("PSV Eindhoven", "PSV Eindhoven"), ("Twente", "FC Twente"), ("Utrecht", "FC Utrecht"),
        ("For Sittard", "Fortuna Sittard"), ("Heracles", "Heracles Almelo"), ("Nijmegen", "NEC Nijmegen"),
        ("Zwolle", "PEC Zwolle"), ("AZ Alkmaar", "AZ Alkmaar"), ("Volendam", "FC Volendam"),
    ],
    "CHAMPIONSHIP": [
        ("Sheffield Weds", "Sheffield Wednesday"), ("Sheffield United", "Sheffield United"),
        ("West Brom", "West Bromwich Albion"), ("QPR", "Queens Park Rangers"), ("Preston", "Preston North End"),
        ("Hull", "Hull City"), ("Ipswich", "Ipswich Town"), ("Coventry", "Coventry City"), ("Wrexham", "Wrexham AFC"),
        ("Bristol City", "Bristol City"),
    ],
    "PRIMEIRA_LIGA": [
        ("Sp Lisbon", "Sporting Lisbon"), ("Porto", "FC Porto"), ("Benfica", "Benfica"), ("Sp Braga", "Braga"),
        ("Guimaraes", "Vitoria Guimaraes"), ("Estrela", "Estrela Amadora"), ("AVS", "AVS Futebol SAD"),
        ("Gil Vicente", "Gil Vicente"),
    ],
    "MLS": [
        ("Atlanta Utd", "Atlanta United FC"), ("Los Angeles Galaxy", "LA Galaxy"), ("Los Angeles FC", "Los Angeles FC"),
        ("DC United", "D.C. United"), ("CF Montreal", "CF Montréal"), ("Inter Miami", "Inter Miami CF"),
        ("New York City", "New York City FC"), ("New York Red Bulls", "New York Red Bulls"),
        ("Seattle Sounders", "Seattle Sounders FC"), ("St. Louis City", "St. Louis City SC"),
        ("Columbus Crew", "Columbus Crew SC"), ("Austin FC", "Austin FC"),
    ],
}


@pytest.mark.parametrize("league", list(PAIRS))
def test_odds_api_names_map_to_football_data(league: str) -> None:
    norm = TeamNameNormalizer()
    fd_names = [fd for fd, _ in PAIRS[league]]
    wrong = {odds: norm.match(odds, fd_names) for fd, odds in PAIRS[league] if norm.match(odds, fd_names) != fd}
    assert not wrong, wrong
