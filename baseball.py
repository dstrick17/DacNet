# pip install pybaseball
import pybaseball
from pybaseball import statcast
from pybaseball import playerid_lookup
import pandas as pd
# Enable caching
pybaseball.cache.enable()
start_date = "2023-06-01"
end_date = "2023-06-07"

player_info = playerid_lookup("Alonso", "Pete")
print(player_info)

# Pete Alonso MLB ID (key_mlbam is 624413)

# Use the statcast function and access the head of the resulting DataFrame
statcast_data = pybaseball.statcast()
print(statcast_data.head(5))

# Fetch data sate
statcast_data = statcast(start_date, end_date)



# Display first few rows
#print(statcast_data.head())
