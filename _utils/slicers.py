from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


# strategy interface
class Slicer(ABC):
    """
    The Context uses this interface to call the slicer strategies defined by concrete
    strategies.
    """

    @abstractmethod
    def apply_slicer(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Strategy context which calls the slicer strategy by concrete strategies.

        Returns
        -------
        Sliced pd.DataFrame.

        """
        pass


# context
class ContextSlicer:
    """
    The Context defines the reference to one of the concrete slicer strategies for pd.DataFrame.
    """

    slice_strategy: Slicer

    def __init__(self, slice_strategy: Slicer = None) -> None:
        """
        The Context accepts a strategy at runtime.
        """

        self._slice_strategy = slice_strategy

    @property
    def slice_strategy(self) -> Slicer:
        """
        The Context maintains a reference to one of the slicer strategy objects.
        """

        return self._slice_strategy

    @slice_strategy.setter
    def slice_strategy(self, slice_strategy: Slicer) -> None:
        """
        Can change the slicer strategy at runtime with a setter
        """

        self._slice_strategy = slice_strategy

    def slicing(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        The Context delegates some work to the Strategy object to the pd.Dataframe
        """

        df = self._slice_strategy.apply_slicer(df, **kwargs)
        return df


# concrete strategies
class InfoSlicer(Slicer):
    """
    Concrete slicer strategy class. Only outputs the info columns of the datasets.
    Note: This can be helpful to do quick concenations on slices of the datasets.
    """

    def apply_slicer(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with information columns

        Returns
        -------
        df : pd.DataFrame
            Sliced DataFrame with only information columns

        """

        info_list = [
            "status_code",
            "status_description",
            "date_sofascore",
            "date_oddsportal",
            "country_sofascore",
            "country_oddsportal",
            "league_sofascore",
            "league_oddsportal",
            "season",
            "round",
            "home_team_sofascore",
            "away_team_sofascore",
            "home_team_short_name",
            "away_team_short_name",
            "home_team_oddsportal",
            "away_team_oddsportal",
            "goal_string",
            "goal_ht_sofascore",
            "goal_ft_sofascore",
            "goal_ht_oddsportal",
            "goal_ft_oddsportal",
            "goal_fullandextratime_sofascore",
            "goal_fulltime_oddsportal",
            "goal_fullandextratime_oddsportal",
            "result_ht_sofascore",
            "result_ft_sofascore",
            "result_ht_oddsportal",
            "result_ft_oddsportal",
            "winner_code_sofascore",
            "result_fulltime_oddsportal",
            "url",
        ]

        return df[info_list]


class StatisticsSlicer(Slicer):
    """
    Concrete slicer strategy class. Slices dataset by sofascore data in 6 categories:
        1. formation
        2. player data
        3. incident data during the match
        4. statistics after match
        5. trend (offensive/defensive) during the match (graph)
        6. votes from user base
    """

    def apply_slicer(
        self,
        df: pd.DataFrame,
        formation: bool = False,
        player_data: bool = False,
        incident: bool = False,
        statistics: bool = False,
        graph: bool = False,
        vote: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with match statistics columns from sofascore
        formation : bool, optional
            Slice dataframe formation data. The default is False.
        player_data : bool, optional
            Slice dataframe player data. The default is False.
        incident : bool, optional
            Slice dataframe incident data. The default is False.
        statistics : bool, optional
            Slice dataframe match statistic data. The default is False.
        graph : bool, optional
            Slice dataframe trend (graph) data. The default is False.
        vote : bool, optional
            Slice dataframe user votes data. The default is False.

        Returns
        -------
        df : pd.DataFrame
            Sliced Dataframe of sofascore match statistic

        """

        sliced_dfs = []
        if formation:
            sliced_dfs.append(df.filter(regex="formation"))
        if player_data:
            sliced_dfs.append(
                df.filter(regex="home_num|away_num|home_player|away_player")
            )
        if incident:
            sliced_dfs.append(df.filter(regex="incident"))
        if statistics:
            sliced_dfs.append(df.filter(regex="stats"))
        if graph:
            sliced_dfs.append(df.filter(regex="minute"))
        if vote:
            sliced_dfs.append(df.filter(regex="vote"))

        if sliced_dfs:
            return pd.concat(sliced_dfs, axis=1, join="inner")
        else:
            return df


class OddsSlicer(Slicer):
    """
    Concrete slicer strategy class. This slices the specified odds data from oddsportal:
    4 types of columns can be sliced:
        1. bookmaker: String of the specified bookmaker (only specified bookmaker are listed)
        2. odds_market: Market where the bet is made (only specified markets are listed)
        3. open_closed: Opening odds or closing odds (only opening or closing odds are listed)
        4. active: The active column of odds is retained
    """

    def apply_slicer(
        self,
        df: pd.DataFrame,
        bookmaker: List[str] = None,
        odds_market: List[str] = None,
        open_closed: List[str] = None,
        active: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns with oddsportal odds data
        bookmaker : List[str], optional
            List with bookmakers to slice. The default is None.
        odds_market : List[str], optional
            List with markets to slice. The default is None.
        open_closed : List[str], optional
            List to slice closed or open odds. The default is None.
        active : bool, optional
            Adds the active column of odds. The default is True.

        Raises
        ------
        ValueError
            Raises value error if bookmaker, market or wrong open/closed string is found.

        Returns
        -------
        df : pd.DataFrame
            Sliced DataFrame with the specified odds data

        """

        bookmaker = [] if bookmaker is None else bookmaker
        odds_market = [] if odds_market is None else odds_market
        open_closed = [] if open_closed is None else open_closed
        if not isinstance(bookmaker, list):
            bookmaker = [bookmaker]
        if not isinstance(odds_market, list):
            odds_market = [odds_market]
        if not isinstance(open_closed, list):
            open_closed = [open_closed]

        allowed_bookmaker = [
            "iFortuna.cz",
            "eFortuna.pl",
            "Tipsport.cz",
            "STS.pl",
            "bet-at-home",
            "Chance.cz",
            "bwin.es",
            "bwin",
            "iFortuna.sk",
            "William Hill",
            "BetVictor",
            "10Bet",
            "ComeOn",
            "Betfair Exchange",
            "Betsafe",
            "Betsson",
            "bet365",
            "Marathonbet",
            "888sport",
            "Unibet",
            "188BET",
            "NordicBet",
            "Vulkan Bet",
            "Winline.ru",
            "Tipsport.sk",
            "bet365.it",
            "1xStavka.ru",
            "1xBet",
            "Marsbet",
            "Planetwin365",
            "Betfair",
            "Unibet.it",
            "Dafabet",
            "Betway",
            "Coolbet",
            "WilliamHill.it",
            "Interwetten",
            "Betfred",
            "bwin.it",
            "Pinnacle",
            "Totolotek.pl",
            "Curebet",
            "VOBET",
            "10x10bet",
            "Lasbet",
            "GGBET",
            "GGBET.ru",
            "Eurobet.it",
            "N1 Bet",
            "bwin.fr",
            "Smarkets",
            "Sportium.es",
            "Matchbook",
            "France Pari",
            "Betclic.fr",
        ]

        allowed_odds_markets = [
            "1x2",
            "BTTS",
            "DC",
            "DNB",
            "HTFT",
            "OU",
            "AH",
            "CS",
        ]

        allowed_open_closed = ["open", "closed"]

        # sanity check the allowed values
        for bm in bookmaker:
            if bm not in allowed_bookmaker:
                raise ValueError(
                    f"Invalid bookmaker: '{bm}'. Allowed bookmakers are {allowed_bookmaker}."
                )

        for om in odds_market:
            if om not in allowed_odds_markets:
                raise ValueError(
                    f"Invalid market: '{om}'. Allowed market are {allowed_odds_markets}."
                )

        for oc in open_closed:
            if oc not in allowed_open_closed:
                raise ValueError(
                    f"Invalid time: '{oc}'. Allowed times are {allowed_open_closed}."
                )
        open_closed = list(
            set(allowed_open_closed) - set(open_closed)
        )  # get opposite of value to drop the open or closed columns and not filter by them (otherwise we lose the active column)

        if bookmaker:  # slice bookmaker
            pattern1 = "|".join(list(map(lambda x: x + "_", bookmaker)))
            df = df.filter(regex=pattern1)
        if odds_market:  # slice odds market
            pattern2 = "|".join(list(map(lambda x: "_" + x + "_", odds_market)))
            df = df.filter(regex=pattern2)
        if open_closed:  # drop open or closed odds
            pattern3 = "|".join(list(map(lambda x: "^(?!.*_" + x + ")", open_closed)))
            df = df.filter(regex=pattern3)
        if not active:  # drop active column
            pattern4 = "^(?!.*_active)"
            df = df.filter(regex=pattern4)
        return df
