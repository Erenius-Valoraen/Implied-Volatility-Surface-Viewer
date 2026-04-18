import glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


option = pd.read_parquet('data/2026-01-02/09-16/options.parquet')
spot_price = pd.read_parquet('data/2026-01-02/09-16/spot.parquet')
spot_price = spot_price[spot_price['instrument'] == 'cash'].close.iloc[0]


option = option[['expiry', 'strike', 'call_close', 'put_close', 'call_open_interest', 'put_open_interest', 'call_timestamp', 'put_timestamp']].dropna()

