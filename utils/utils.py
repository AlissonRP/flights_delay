import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def grouped_delay2(df, var, type="bar", order=True, metric="mean"):
    """
    OLD
    """
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
        fig = px.area(data, x=var, y="PROPORCAO", title="Life expectancy in Canada")
        fig.add_hline(
            y=prop_mean, line_dash="dot", annotation_text=metric, line_color="red"
        )
        return fig.update_layout(showlegend=False)


def grouped_delay(df, var, type="bar", order=True, metric="mean"):
    """
    df: DataFrame
    var: Variable
    type: Type of plot, default is bar
    order: To order the columns of DataFrame
    metric: metric to plot in horizontal line
    """
    data = (
        df.groupby(var, as_index=False)["ATRASO"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    data = data.assign(PROPORCAO=data["sum"] / data["count"])
    prop_metric = data["PROPORCAO"].agg(metric)
    if order == True:
        data = data.sort_values(["PROPORCAO"], ascending=False)
    if type == "bar":
        sns.barplot(data=data, x=var, y="PROPORCAO")
        plt.axhline(prop_metric, color="r", linestyle="--")
        plt.annotate(metric, xy=(5, 0.385))
        plt.title("Proporção de Atrasos")
    else:
        sns.lineplot(data=data, x=var, y="PROPORCAO")
        plt.axhline(prop_metric, color="r", linestyle="--")
        plt.annotate(metric, xy=(5, 0.385))
        plt.title("Proporção de Atrasos")
        plt.legend(loc="upper right")


def total_bar2(df, var, order=True):
    """
    OLD
    """
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


def total_bar(df, var, order=True):
    """
    df: DataFrame
    var: Variable
    """
    data = df.groupby([var], as_index=False).size()
    if order == True:
        data = data.sort_values(["size"], ascending=False)
    sns.barplot(data=data, x=var, y="size")
    plt.title("Total da variável " + var)


def create_proportion(df, var):
    """
    df: DataFrame
    var: Variable
    """
    data = (
        df.groupby(var, as_index=False)["ATRASO"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    data = data.assign(PROPORCAO_ATRASO=data["sum"] / data["count"])
    return data
