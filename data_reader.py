import glob
import pandas as pd
import plotly.graph_objects as go
from black_scholes import BlackScholes, implied_volatility
import numpy as np

# 🔹 Load ALL time snapshots
base_path = "data/2026-01-02/*/options.parquet"
files = sorted(glob.glob(base_path))

all_data = []
spot_map = {}  # store spot per timestamp

r = 0.06  # risk-free rate

for file in files:
    timestamp_str = file.split("/")[-2]
    timestamp = pd.Timestamp(f"2026-01-02 {timestamp_str.replace('-', ':')}")

    options = pd.read_parquet(file)
    spot = pd.read_parquet(file.replace("options.parquet", "spot.parquet"))
    spot_price = spot[spot['instrument'] == 'cash'].close.iloc[0]

    spot_map[timestamp] = spot_price

    options = options[['expiry', 'strike', 'call_close', 'put_close',
                       'call_open_interest', 'put_open_interest']].dropna()

    ivs = []

    for _, row in options.iterrows():
        time_to_expiry = ((row.expiry - timestamp).total_seconds() / (24 * 3600)) / 365

        if time_to_expiry <= 0:
            ivs.append(np.nan)
            continue

        try:
            if row.strike > spot_price:
                iv = implied_volatility(
                    row.call_close,
                    spot_price,
                    row.strike,
                    time_to_expiry,
                    r,
                    'call'
                ) * 100
            else:
                iv = implied_volatility(
                    row.put_close,
                    spot_price,
                    row.strike,
                    time_to_expiry,
                    r,
                    'put'
                ) * 100
        except:
            iv = np.nan

        ivs.append(iv)

    options['iv'] = ivs
    options['timestamp'] = timestamp

    all_data.append(options)

# 🔹 Combine all timestamps
options_all = pd.concat(all_data)

# 🔹 Get unique timestamps
timestamps = sorted(options_all['timestamp'].unique())

frames = []

for ts in timestamps:
    df = options_all[options_all['timestamp'] == ts].copy()
    spot_price = spot_map[ts]

    df["days_to_expiry"] = (df["expiry"] - ts).dt.days

    pivot = df.pivot_table(
        values="iv",
        index="days_to_expiry",
        columns="strike"
    )

    pivot = pivot.interpolate(axis=1).interpolate(axis=0)

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # 🔹 Find ATM strike
    strikes = np.array(pivot.columns)
    atm_strike = strikes[np.argmin(np.abs(strikes - spot_price))]

    # 🔹 Create ATM line (vertical across expiries)
    atm_x = [atm_strike] * len(pivot.index)
    atm_y = pivot.index
    atm_z = [np.nanmax(Z)] * len(pivot.index)  # lift above surface

    frames.append(
        go.Frame(
            data=[
                go.Surface(x=X, y=Y, z=Z),
                go.Scatter3d(
                    x=atm_x,
                    y=atm_y,
                    z=atm_z,
                    mode='lines',
                    line=dict(color='red', width=6),
                    name='ATM'
                )
            ],
            name=str(ts.time())
        )
    )

# 🔹 Initial frame
first_ts = timestamps[0]
first_df = options_all[options_all['timestamp'] == first_ts].copy()
spot_price = spot_map[first_ts]

first_df["days_to_expiry"] = (first_df["expiry"] - first_ts).dt.days

pivot = first_df.pivot_table(
    values="iv",
    index="days_to_expiry",
    columns="strike"
)

pivot = pivot.interpolate(axis=1).interpolate(axis=0)

X, Y = np.meshgrid(pivot.columns, pivot.index)
Z = pivot.values

# ATM for first frame
strikes = np.array(pivot.columns)
atm_strike = strikes[np.argmin(np.abs(strikes - spot_price))]

atm_x = [atm_strike] * len(pivot.index)
atm_y = pivot.index
atm_z = [np.nanmax(Z)] * len(pivot.index)

# 🔹 Create figure
fig = go.Figure(
    data=[
        go.Surface(x=X, y=Y, z=Z),
        go.Scatter3d(
            x=atm_x,
            y=atm_y,
            z=atm_z,
            mode='lines',
            line=dict(color='red', width=6),
            name='ATM'
        )
    ],
    frames=frames
)

# 🔹 Slider + controls
fig.update_layout(
    title="IV Surface (Time Evolution) with ATM Highlight",
    scene=dict(
        xaxis_title="Strike",
        yaxis_title="Days to Expiry",
        zaxis_title="IV"
    ),
    sliders=[{
        "steps": [
            {
                "args": [[str(ts.time())],
                         {"frame": {"duration": 300, "redraw": True},
                          "mode": "immediate"}],
                "label": str(ts.time()),
                "method": "animate"
            }
            for ts in timestamps
        ]
    }],
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "Play",
             "method": "animate",
             "args": [None, {"frame": {"duration": 300, "redraw": True}}]},
            {"label": "Pause",
             "method": "animate",
             "args": [[None], {"frame": {"duration": 0}}]}
        ]
    }]
)

fig.show()