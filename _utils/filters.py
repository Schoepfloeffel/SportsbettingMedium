from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


# strategy interface
class Filter(ABC):
    """
    The Context uses this interface to call the filter strategies defined by concrete
    strategies.
    """

    @abstractmethod
    def apply_filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Strategy context which calls the filter strategy by concrete strategies.

        Returns
        -------
        Filtered pd.DataFrame.

        """
        pass


# context
class ContextFilter:
    """
    The Context defines the reference to one of the concrete filter strategies for pd.DataFrame.
    """

    filter_strategy: Filter

    def __init__(self, filter_strategy: Filter = None) -> None:
        """
        The Context accepts a strategy at runtime.
        """

        self._filter_strategy = filter_strategy

    @property
    def filter_strategy(self) -> Filter:
        """
        The Context maintains a reference to one of the filter strategy objects.
        """

        return self._filter_strategy

    @filter_strategy.setter
    def filter_strategy(self, filter_strategy: Filter) -> None:
        """
        Can change the filter strategy at runtime with a setter
        """

        self._filter_strategy = filter_strategy

    def filtering(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        The Context delegates some work to the Strategy object to the pd.Dataframe
        """

        df = self._filter_strategy.apply_filter(df, **kwargs)
        return df


# concrete strategies
class CountryFilter(Filter):
    """
    Concrete filter strategy class. Filters dataset by country in two columns ['country_sofascore', 'country_oddsportal']
    """

    def apply_filter(self, df: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns ['country_sofascore', 'country_oddsportal']
        countries : List[str]
            List of countries to filter for (e.g., ["Brazil", "Chile"])

        Returns
        -------
        df: pd.Dataframe
            filtered by countries on list
        """

        return df[
            df[["country_sofascore", "country_oddsportal"]].isin(countries).any(axis=1)
        ]


class DateFilter(Filter):
    """
    Concrete filter strategy class. Filters dataset by date from column ['date_sofascore'].
    It is advised to primarly use the date from sofascore.
    Otherwise you can fallback on ['date_oddsportal'], but experience showed it is more error prone.
    """

    def apply_filter(
        self,
        df: pd.DataFrame,
        date_start: pd.Timestamp,
        date_end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with column ['date_sofascore']
        date_start : List[str]
            pd.Timestamp of start date < filtered matches
        date_start : List[str]
            pd.Timestamp of end date > filtered matches

        Returns
        -------
        df: pd.Dataframe
            filtered by date range
        """

        return df[df["date_sofascore"].between(date_start, date_end)]


class SeasonFilter(Filter):
    """
    Concrete filter strategy class. Filters dataset by year from column ['season'].
    This column is unique to sofascore data. If sofadata is missing, the value for season entry is 'None'.
    """

    def apply_filter(
        self,
        df: pd.DataFrame,
        year: int,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with column ['season']
        year : int
            season year to filter for

        Returns
        -------
        df: pd.Dataframe
            filtered by season
        """

        return df[df["season"].str.contains(str(year), na=False)]


class StatusFilter(Filter):
    """
    Concrete filter strategy class. Filters dataset by status from column ['status_code'].
    This column is unique to sofascore data. If sofadata is missing, the value for season entry is 'NaN'.

    If the status code is not 100, then the goal data is wrong for fulltime.
    This needs to be adjusted by halftime and fulltime goals from columns ['goal_ht_*', 'goal_ft_*'] if necessary
    """

    def apply_filter(self, df: pd.DataFrame, status_list: List[int]) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with column ['season']
        status_list : List[int]
            status codes to filter for

        Returns
        -------
        df: pd.Dataframe
            filtered by season
        """

        allowed_values = [
            60,  # Postponed
            70,  # Canceled
            80,  # Interrupted
            100,  # Ended (Ended with normal play time)
            110,  # AET (overtime)
            120,  # AP (overtime with penalties)
        ]

        for status in status_list:
            if status not in allowed_values:
                raise ValueError(
                    f"Invalid status value: '{status}'. Allowed values are {allowed_values}."
                )

        return df[df.status_code.isin(status_list)]


class StatisticsFilter(Filter):
    """
    Concrete filter strategy class. Filters dataset by availabilty of sofascore data in 6 categories:
        1. formation
        2. player data
        3. incident data during the match
        4. statistics after match
        5. trend (offensive/defensive) during the match (graph)
        6. votes from user base
    Only one column is checked by category. If the column contains data, it indicates that data in this category is present.
    Before filtering, 'nan' strings are forced to np.nan type.

    Note: Some columns exhibit 'nan' and None entries, both rows with those entires will be dropped.
    """

    def apply_filter(
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
            Dataframe with information columns from sofascore
        formation : bool, optional
            Filters by availability of formation data. The default is False.
        player_data : bool, optional
            Filters by availability of player data. The default is False.
        incident : bool, optional
            Filters by availability of incident data. The default is False.
        statistics : bool, optional
            Filters by availability of match statistic data. The default is False.
        graph : bool, optional
            Filters by availability of formation trend (graph) data. The default is False.
        vote : bool, optional
            Filters by availability of user votes data. The default is False.

        Returns
        -------
        df : pd.DataFrame
            filtered by sofascore information

        """

        if formation:
            df = df[df["home_lineup_formation"].notna()]
        if player_data:
            df = df[df["home_num_players"].notna()]
        if incident:
            df = df[df["incident_incidentType_0"].notna()]
        if statistics:
            df = df[df["stats_all_tvdata_corner_kicks_home"].notna()]
        if graph:
            df = df[df["minute_1"].notna()]
        if vote:
            df = df[df["vote1"].notna()]
        return df


class OddsFilter(Filter):
    """
    Concrete filter strategy class. This filter the availabilty of odds data from oddsportal:
    4 types of columns can be filtered. "Active" column indicates if the bet offer was active/valid:
        1. bookmaker: String of the specified bookmaker (only available entries for specified bookmaker are listed)
        2. odds_market: Market where the bet is made (only available entries for specified markets are listed)
        3. open_closed: Opening odds or closing odds (only available entries for opening or closing odds are listed)
        4. active: Odds are active and the bet can be made for this bookmaker (only availabe entries for valid bets are listed)
    """

    def apply_filter(
        self,
        df: pd.DataFrame,
        bookmaker: List[str] = None,
        odds_market: List[str] = None,
        open_closed: List[str] = None,
        active: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns with oddsportal odds data
        bookmaker : List[str], optional
            List with bookmakers to filter. The default is None.
        odds_market : List[str], optional
            List with markets to filter. The default is None.
        open_closed : List[str], optional
            List to filter closed or open odds. The default is None.
        active : bool, optional
            Checks if the odds were available and active for betting. The default is False.

        Raises
        ------
        ValueError
            Raises value error if bookmaker, market or wrong open/closed string is found.

        Returns
        -------
        df : pd.DataFrame
            Filtered df with the available odds data

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
                    f"Invalid market: '{oc}'. Allowed market are {allowed_open_closed}."
                )
        open_closed = list(
            set(allowed_open_closed) - set(open_closed)
        )  # get opposite of value to drop the open or closed columns and not filter by them (otherwise we lose the active column)

        # filter dataset by bookmaker, markets and opening/closing odds availabilty
        tmp_df = df
        if bookmaker:
            pattern1 = "|".join(list(map(lambda x: x + "_", bookmaker)))
            tmp_df = tmp_df.filter(regex=pattern1)
        if odds_market:
            pattern2 = "|".join(list(map(lambda x: "_" + x + "_", odds_market)))
            tmp_df = tmp_df.filter(regex=pattern2)
        if open_closed:
            pattern3 = "|".join(list(map(lambda x: "^(?!.*_" + x + ")", open_closed)))
            tmp_df = tmp_df.filter(regex=pattern3)

        # filter dataset by active odds (0 and NaN is dropped)
        if active:
            tmp_df = tmp_df.replace(0, np.nan)  # replace 0 (inactive) with NaN
        else:
            tmp_df = tmp_df.filter(regex="^(?!.*_active)")  # drop active column

        return df[
            tmp_df.notna().all(axis=1)
        ]  # filter all NaN after we checked the NaNs from sliced dataframe
