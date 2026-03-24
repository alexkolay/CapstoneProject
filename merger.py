import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import json

#NOTE Spacetrack contains the same satellites as Celestrak PLUS decayed satellites (fell out of orbit).
#Celestrak only contains active satellites. All Celestrak satellites are in Spacetrack

with open('spacetrack_starlink.json', 'r') as f:
    spacetrack_data = json.load(f)

spacetrack_df = pd.DataFrame(spacetrack_data)
celestrak_df = pd.read_csv('celestrak_starlink.csv')

spacetrack_df['NORAD_CAT_ID'] = spacetrack_df['NORAD_CAT_ID'].astype('int64')

active_satellites = pd.merge(spacetrack_df, celestrak_df, on='NORAD_CAT_ID', how='inner')
#Active satellites are satellites that are still in orbit, MOST useful for our models

active_with_decay = active_satellites[active_satellites['DECAY_DATE'].notna()]
#If theres anything here it means theres either an issue or just delay between when we scraped the datasets

inactive_satellites = spacetrack_df[spacetrack_df['DECAY_DATE'].notna()]
#Satelittes that are decayed. Not present in celestrak, and probably not useful for our models




