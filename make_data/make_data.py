#%%
import pandas as pd
import numpy as np


# %%

#%%
df = pd.read_csv("../data/flights.csv")
df = df[df["ARRIVAL_DELAY"].notna()]


#%%
df.drop(
    [
        "YEAR",
        "FLIGHT_NUMBER",
        "TAIL_NUMBER",
        "TAXI_OUT",
        "CANCELLED",
        "CANCELLATION_REASON",
    ],
    axis=1,
)

#%%
df["ATRASO"] = np.where(df["ARRIVAL_DELAY"] > 0, 1, 0)
#%%
df.to_csv("../data/new_flights.csv")