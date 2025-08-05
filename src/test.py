import nfl_data_py as nfl
import pandas as pd

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns

#weekly_data = nfl.import_weekly_data([2024])

pbp = nfl.import_pbp_data([2024, 2023])
print(pbp.head(20))