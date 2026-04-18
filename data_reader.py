import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from black_scholes import implied_volatility
import numpy as np

# 🔹 Load data
base_path = "data/2026-01-02/*/options.parquet"
files = sorted(glob.glob(base_path))

all_data = []
spot_series = []

r = 0.06

for file in files:
    timestamp_str = file.split("/")[-2]
    timestamp = pd.Timestamp(f"2026-01-02 {timestamp_str.replace('-', ':')}")

    options = pd.read_parquet(file)
    spot = pd.read_parquet(file.replace("options.parquet", "spot.parquet"))
    spot_price = spot[spot['instrument'] == 'cash'].close.iloc[0]

    spot_series.append((timestamp, spot_price))

    options = options[['expiry', 'strike', 'call_close', 'put_close']].dropna()

    ivs = []
    for _, row in options.iterrows():
        T = ((row.expiry - timestamp).total_seconds() / (24 * 3600)) / 365

        if T <= 0:
            ivs.append(np.nan)
            continue

        try:
            if row.strike > spot_price:
                iv = implied_volatility(row.call_close, spot_price, row.strike, T, r, 'call') * 100
            else:
                iv = implied_volatility(row.put_close, spot_price, row.strike, T, r, 'put') * 100
        except:
            iv = np.nan

        ivs.append(iv)

    options['iv'] = ivs
    options['timestamp'] = timestamp

    all_data.append(options)

options_all = pd.concat(all_data)
spot_df = pd.DataFrame(spot_series, columns=["timestamp", "spot"])

timestamps = sorted(options_all['timestamp'].unique())

# 🔹 Create frames
frames = []

for ts in timestamps:
    df = options_all[options_all['timestamp'] == ts].copy()
    df["days_to_expiry"] = (df["expiry"] - ts).dt.days

    pivot = df.pivot_table(values="iv", index="days_to_expiry", columns="strike")
    pivot = pivot.interpolate(axis=1).interpolate(axis=0)

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # Spot data up to this time
    spot_subset = spot_df[spot_df["timestamp"] <= ts]

    frames.append(
        go.Frame(
            data=[
                go.Surface(x=X, y=Y, z=Z),  # surface
                go.Scatter(                  # spot line
                    x=spot_subset["timestamp"],
                    y=spot_subset["spot"],
                    mode="lines",
                    name="Spot Price"
                )
            ],
            name=str(ts.time())
        )
    )

# 🔹 Initial data
first_ts = timestamps[0]
df = options_all[options_all['timestamp'] == first_ts].copy()
df["days_to_expiry"] = (df["expiry"] - first_ts).dt.days

pivot = df.pivot_table(values="iv", index="days_to_expiry", columns="strike")
pivot = pivot.interpolate(axis=1).interpolate(axis=0)

X, Y = np.meshgrid(pivot.columns, pivot.index)
Z = pivot.values

# 🔹 Create subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "xy"}]],
    column_widths=[0.7, 0.3],
    subplot_titles=("IV Surface", "Spot Price")
)

# Initial traces
fig.add_trace(go.Surface(x=X, y=Y, z=Z), row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=[first_ts],
        y=[spot_df.iloc[0]["spot"]],
        mode="lines",
        name="Spot Price"
    ),
    row=1, col=2
)

# 🔹 Attach frames
fig.frames = frames

# 🔹 Layout
fig.update_layout(
    title="IV Surface + Spot Price",

    scene=dict(
        xaxis_title="Strike",
        yaxis_title="Days to Expiry",
        zaxis_title="IV"
    ),

    # 🔹 Correct way to set subplot axis titles
    xaxis2=dict(title="Time"),
    yaxis2=dict(title="Spot Price"),

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
             "args": [None, {"frame": {"duration": 300}}]},
            {"label": "Pause",
             "method": "animate",
             "args": [[None], {"frame": {"duration": 0}}]}
        ]
    }]
)

fig.show()