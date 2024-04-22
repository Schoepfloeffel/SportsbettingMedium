from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    COUNTRY: tuple[str] = (
        "Africa",
        "Albania",
        "Algeria",
        "Andorra",
        "Angola",
        "Argentina",
        "Armenia",
        "Aruba",
        "Asia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Bolivia",
        "Bosnia",
        "Brazil",
        "Bulgaria",
        "Burundi",
        "Cambodia",
        "Cameroon",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Costa Rica",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "DR Congo",
        "Denmark",
        "Ecuador",
        "Egypt",
        "El Salvador",
        "England",
        "Estonia",
        "Ethiopia",
        "Europe",
        "Faroe Islands",
        "Fiji",
        "Finland",
        "France",
        "Gambia",
        "Georgia",
        "Germany",
        "Ghana",
        "Gibraltar",
        "Greece",
        "Guatemala",
        "Honduras",
        "Hong Kong",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iran",
        "Ireland",
        "Israel",
        "Italy",
        "Ivory Coast",
        "Jamaica",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kenya",
        "Kosovo",
        "Kuwait",
        "Latvia",
        "Lebanon",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Malaysia",
        "Malta",
        "Mauritius",
        "Mexico",
        "Moldova",
        "Mongolia",
        "Montenegro",
        "Morocco",
        "Mozambique",
        "Myanmar",
        "Netherlands",
        "New Zealand",
        "Nicaragua",
        "Nigeria",
        "North & Central America",
        "North Macedonia",
        "Northern Ireland",
        "Norway",
        "Oman",
        "Oceania",
        "Pakistan",
        "Palestine",
        "Panama",
        "Paraguay",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Russia",
        "Rwanda",
        "San Marino",
        "Saudi Arabia",
        "Scotland",
        "Senegal",
        "Serbia",
        "Singapore",
        "Slovakia",
        "Slovenia",
        "South America",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Syria",
        "Tajikistan",
        "Tanzania",
        "Thailand",
        "Trinidad and Tobago",
        "Tunisia",
        "Turkey",
        "Turkmenistan",
        "USA",
        "Uganda",
        "Ukraine",
        "United Arab Emirates",
        "Uruguay",
        "Uzbekistan",
        "Venezuela",
        "Vietnam",
        "Wales",
        "World",
        "Zambia",
        "Zimbabwe",
    )

    def __repr__(self) -> str:
        return f"Countries in dataset: {self.COUNTRY}"


@dataclass(frozen=True)
class Bookie:
    BOOKIE: tuple[str] = (
        "10Bet",
        "10x10bet",
        "188BET",
        "1xBet",
        "1xStavka.ru",
        "888sport",
        "bet-at-home",
        "bet365",
        "bet365.it",
        "Betclic.fr",
        "Betfair",
        "Betfair Exchange",
        "Betfred",
        "Betsafe",
        "Betsson",
        "BetVictor",
        "Betway",
        "bwin",
        "bwin.es",
        "bwin.fr",
        "bwin.it",
        "Chance.cz",
        "ComeOn",
        "Coolbet",
        "Curebet",
        "Dafabet",
        "eFortuna.pl",
        "Eurobet.it",
        "France Pari",
        "GGBET",
        "GGBET.ru",
        "iFortuna.cz",
        "iFortuna.sk",
        "Interwetten",
        "Lasbet",
        "Marathonbet",
        "Marsbet",
        "Matchbook",
        "N1 Bet",
        "NordicBet",
        "Pinnacle",
        "Planetwin365",
        "Smarkets",
        "Sportium.es",
        "STS.pl",
        "Tipsport.cz",
        "Tipsport.sk",
        "Totolotek.pl",
        "Unibet",
        "Unibet.it",
        "VOBET",
        "Vulkan Bet",
        "William Hill",
        "WilliamHill.it",
        "Winline.ru",
    )

    def __repr__(self) -> str:
        return f"Allowed bookmaker: {self.BOOKIE}"


@dataclass(frozen=True)
class Market:
    MARKET: tuple[str] = (
        "1x2",  # 3-way win
        "AH",  #  asian handicap
        "BTTS",  # both teams to score
        "CS",  # correct score
        "DC",  # double chance
        "DNB",  # draw no bet
        "HTFT",  # 1./2. half
        "OU",  # over under
    )

    def __repr__(self) -> str:
        return f"Allowed markets: {self.MARKET}"