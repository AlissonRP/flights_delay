#%%
import pandas as pd
import numpy as np
import plotly.express as px
from utils.utils import grouped_delay, total_bar
import seaborn as sns
import matplotlib.pyplot as plt

# %%

#%%
df = pd.read_csv("data/flights.csv")
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
df.to_csv("new_flight.csv")
#%%
df.groupby(["ATRASO"], as_index=False).size()
# %%
df = pd.read_csv("data/new_flight.csv").sample(n=100000)

#%% ATRASOS
teste = df.groupby(["ATRASO"], as_index=False).size()

#%%
teste["proporcao"] = teste["size"] / teste["size"].sum()


#%%
px.pie(teste, values="size", names="ATRASO", title="Proporçao de Atrasos")


## MODOS DE ATRASO
# Substituir qualquier valor maior que zero por 1, e torna essas variaveis binarias
#%%
df.iloc[:, 27:33] = df.iloc[:, 27:33].fillna(0)

#%%
def grouped_delay(df, var, type="bar", order=True, metric="mean"):
    data = df.groupby([var], as_index=False)["ATRASO"].sum()
    data = data.assign(PROPORCAO=data["ATRASO"] / data["ATRASO"].sum())
    prop_mean = data["PROPORCAO"].agg(metric)
    if type == "bar":
        fig = px.bar(
            data, x=var, y="PROPORCAO", title="Proporçao de Atrasos", color=var
        )
        fig.add_hline(
            y=prop_mean, line_dash="dot", annotation_text=metric, line_color="red"
        )
        if order == True:
            return fig.update_layout(
                barmode="stack",
                xaxis={"categoryorder": "total descending"},
                showlegend=False,
            )
        else:
            return fig.update_layout(showlegend=False)
    else:
        fig = px.area(data, x=var, y="PROPORCAO", title="")
        fig.add_hline(
            y=prop_mean, line_dash="dot", annotation_text=metric, line_color="red"
        )
        return fig.update_layout(showlegend=False)


#%%
def grouped_delay(df, var, type="bar", order=True, metric="mean"):
    data = (
        df.groupby(var, as_index=False)["ATRASO"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    data = data.assign(PROPORCAO=data["sum"] / data["count"])
    prop_metric = data["PROPORCAO"].agg("mean")
    if order == True:
        data = data.sort_values(["PROPORCAO"], ascending=False)
    if type == "bar":
        sns.barplot(data=data, x=var, y="PROPORCAO")
        plt.axhline(prop_metric, color="r")
        plt.title("Proporção de Atrasos")
    else:
        sns.lineplot(data=data, x=var, y="PROPORCAO")
        plt.axhline(prop_metric, color="r")
        plt.title("Proporção de Atrasos")
        plt.legend(loc="upper right")


#%%


def total_bar(df, var, order=True):
    data = df.groupby([var], as_index=False).size()
    fig = px.bar(data, x=var, y="size", title="Total da variável " + var, color=var)
    if order == True:
        return fig.update_layout(
            barmode="stack",
            xaxis={"categoryorder": "total descending"},
            showlegend=False,
        )
    else:
        return fig.update_layout(showlegend=False)


#%%
def total_bar2(df, var, order=True):
    data = df.groupby([var], as_index=False).size()
    if order == True:
        data = data.sort_values(["size"], ascending=False)
    sns.barplot(data=data, x=var, y="size")
    plt.title("Total da variável " + var)


#%%
total_bar2(df, "AIRLINE")


#%%
grouped_delay(df, "AIRLINE")

#%%
df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].map(
    {
        1: "Segunda",
        2: "Terca",
        3: "Quarta",
        4: "Quinta",
        5: "Sexta",
        6: "Sabado",
        7: "Domingo",
    }
)


#%%
total_bar(df, "MONTH")
data = df.groupby("MONTH", as_index=False).size()
px.area(data, x="MONTH", y="size", title="Life expectancy in Canada")

#%%
fig = grouped_delay(df, "MONTH")
fig.add_vrect(
    x0="8.5",
    x1="11.5",
    annotation_text="declinio",
    annotation_position="top left",
    annotation=dict(font_size=20, font_family="Times New Roman"),
    fillcolor="red",
    opacity=0.25,
    line_width=0,
)
#%%

fig = grouped_delay(df, "DAY_OF_WEEK", type="line")
fig.add_vrect(
    x0="5",
    x1="6",
    annotation_text="Fim de semana",
    annotation_position="top left",
    annotation=dict(font_size=16, font_family="Times New Roman"),
    fillcolor="red",
    opacity=0.25,
    line_width=0,
)

# menos atrasos nos fim de semana
#
#%%
total_bar2(df, 'DAY')
#%%

grouped_delay(df, "DAY")

#
#%% as
total_bar2(df, 'DAY_OF_WEEK')
# %%

grouped_delay(df, "DAY_OF_WEEK")


#%%


total_bar2(df, 'MONTH')
# %%

grouped_delay(df, "MONTH")


#%%
total_bar2(df, "AIRLINE")

#%%
grouped_delay(df, "AIRLINE")


#%%


#%%
dist = df[["DISTANCE"]]
df["CAT_DISTANCE"] = np.where(
    dist <= 250,
    "0",
    np.where(
        (250 < dist) & (dist <= 500),
        "1",
        np.where(
            (500 < dist) & (dist <= 1000),
            "2",
            np.where(
                (1000 < dist) & (dist <= 1200),
                "3",
                np.where((1200 < dist) & (dist <= 2000), "4", "5"),
            ),
        ),
    ),
).flatten()
###

#%%

total_bar2(df, "CAT_DISTANCE")
#%%

grouped_delay(df, "CAT_DISTANCE")


# Nota-se que distancias de 300 a 600 tem muito mais atrasos,
# e conforme  a distancia aumenta a quantidade de atrasos diminui
# """
#%%


def create_proportion(df, var):
    data = (
        df.groupby(var, as_index=False)["ATRASO"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    data = data.assign(PROPORCAO_ATRASO=data["sum"] / data["count"])
    return data


#%%
df.groupby("CAT_DISTANCE").size()

#%%
origin_airport = create_proportion(df, "ORIGIN_AIRPORT")
origin_airport = origin_airport[origin_airport["count"] > 10]
#%%
prop = origin_airport["PROPORCAO_ATRASO"]
origin_airport["HISTORICO_ATRASO"] = np.where(
    prop < 0.15, "OK", np.where((0.15 < prop) & (prop < 0.3), "razoavel", "preocupante")
)
#%%


cia_atrasos = create_proportion(df, "AIRLINE")

#%%
np.where(cia_atrasos['PROPORCAO_ATRASO'] > cia_atrasos['PROPORCAO_ATRASO'].mean(), "preocupante", "ok")
#%%

origin_airport["ATRASO_PROP"].mean()
#%%
grouped_delay(df, "ORIGIN_AIRPORT")


#%%


#%%

px.histogram(df, x="DISTANCE")


#%%

px.box(df, y="DEPARTURE_DELAY")


#%%
df[df["DEPARTURE_DELAY"] > 1058]


#%%
### ANÁLISE DA VARIÁVEIS DE TEMPO

df["SCHEDULED_ARRIVAL"] = pd.to_datetime(
    df["SCHEDULED_ARRIVAL"].astype(str).str.zfill(4), format="%H%M"
).dt.strftime("%H")


#%%


total_bar(df, "SCHEDULED_ARRIVAL", order=False)
#%%
grouped_delay(df, "SCHEDULED_ARRIVAL", order=False, metric="median")


#%%
df["END_WEEK"] = np.where(df["DAY_OF_WEEK"] >= 6, 1, 0)

##### MODELAGEEEM
#%%
df_model = df[
    [
        "END_WEEK",
        "CAT_DISTANCE",
        "DISTANCE",
        "SCHEDULED_ARRIVAL",
        "DEPARTURE_DELAY",
        "MONTH",
        "ATRASO",
    ]
]
df_model[["CAT_DISTANCE", "SCHEDULED_ARRIVAL"]] = df[
    ["CAT_DISTANCE", "SCHEDULED_ARRIVAL"]
].apply(pd.to_numeric)

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
#%%
X = df_model.loc[:, df_model.columns != "ATRASO"]  # Features
y = df_model["ATRASO"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#%%
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#%%


from sklearn import metrics
from sklearn.metrics import accuracy_score

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#%%
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)

#%%
accuracy = accuracy_score(y_test, model.predict(X_test))

#%%
df_model.isna().info()

#%%

data = (
    df.groupby("DAY", as_index=False)["ATRASO"]
    .agg(["sum", "count"])
    .reset_index()
    .sort_values("count", ascending=False)
)
data = data.assign(PROPORCAO=origin_airport["sum"] / origin_airport["count"])
