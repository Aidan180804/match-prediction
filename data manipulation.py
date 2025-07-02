import pandas as pd
from skimage.metrics import mean_squared_error

#import dataframes
results = pd.read_csv(r"C:\Users\aidan\OneDrive - The University of Nottingham\Documents\kaggle\prem\results (17-18).csv")
stats = pd.read_csv(r"C:\Users\aidan\OneDrive - The University of Nottingham\Documents\kaggle\prem\stats.csv")

#remove unwanted columns
results = pd.DataFrame(results.iloc[:380,[0,1,4]])
stats = pd.DataFrame(stats.drop(columns = ['wins','losses','season']).iloc[220:,:])

#change stats to per game format
numerical = stats.select_dtypes(include = 'number').columns
stats[numerical] = stats[numerical]/38

stats_columns = stats.columns

#change index names and split into home and away stats
stats = stats.set_index('team')
results_home_stats =results.set_index('home_team')
results_away_stats = results.set_index('away_team')

#input stats data into results dataframes
results_home_stats[stats.columns] = stats[stats.columns]
results_away_stats[stats.columns] = stats[stats.columns]

#change column names to home and away specific
results_home_stats = results_home_stats.add_prefix('home_')
results_away_stats = results_away_stats.add_prefix('away_')
results_home_stats = results_home_stats.rename(columns={"home_away_team": "away_team",'home_result':'result'})
results_away_stats = results_away_stats.rename(columns={"away_home_team": "home_team",'away_result':'result'})

#remove indices
results_home_stats = results_home_stats.reset_index(drop=True)
results_away_stats = results_away_stats.reset_index(drop=True)

#combine
combined_stats = pd.concat([results_home_stats, results_away_stats], axis=1)

#rearrange home team column to beginning
home_team = combined_stats.pop('home_team')
combined_stats.insert(1,'home_team',home_team)

#remove duplicate result column
combined_stats = combined_stats.loc[:, ~combined_stats.columns.duplicated()]
