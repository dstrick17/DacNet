# pip install pybaseball
import pybaseball
from pybaseball import statcast
from pybaseball import playerid_lookup
import pandas as pd
# Enable caching
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pybaseball.cache.enable()

player_id = 624413  # Pete Alonso MLB ID

# player_info = playerid_lookup("Alonso", "Pete")
# print(player_info)

# statcast_data = pybaseball.statcast() # Stat cast is by pitch

season2024_batting_stats = pybaseball.batting_stats_bref(2024)
mets_batting_stats = season2024_batting_stats[(season2024_batting_stats['Lev'] == 'Maj-NL') & (season2024_batting_stats['Tm'] == 'New York')]
print(mets_batting_stats)








