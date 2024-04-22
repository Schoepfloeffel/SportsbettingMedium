import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_heatmap_goals(df: pd.DataFrame):
    """
    Plot heatmap of goal distribution based on odds DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the relevant data.

    Returns:
    None

    """
    goals = pd.DataFrame()
    goals[["Home", "Away"]] = df["goal_fullandextratime_sofascore"].str.split(
        ":", expand=True
    )
    goals_count = pd.crosstab(goals["Home"], goals["Away"]).iloc[
        :7, :7
    ]  # slice to 5 goals max
    goals_perc = (
        pd.crosstab(goals["Home"], goals["Away"], normalize="all").iloc[:7, :7] * 100
    )
    annot = (
        goals_count.astype(str) + "\n" + "(" + goals_perc.round(2).astype(str) + "%)"
    )
    # create heatmap
    axes = sns.heatmap(
        goals_count,
        annot=annot,
        linewidth=0.5,
        fmt="",
        cmap=sns.light_palette("xkcd:copper", 8),
        cbar=False,
        annot_kws={"ha": "center", "va": "center"},
    )
    axes.set_title(
        f"Goal distribution - Result after 90 min (n={goals_count.sum().sum()}) - in league {df['league_sofascore'].unique()}",
        fontsize=12,
    )
    # create patches to visualize home win, draw, away win
    colors = [(1, 0, 0, 0.5), (0, 0, 0, 1), (0, 0, 1, 0.5)]
    rows = len(goals_perc.index)
    cols = len(goals_perc.columns)
    for row in range(rows):
        for col in range(cols):
            home_goal = row
            away_goal = col
            # add patches, position (0,0) is bottom left
            if home_goal > away_goal:  # home win
                axes.add_patch(
                    Rectangle(
                        (rows - row - 1, cols - col - 1),
                        1,
                        1,
                        fill=False,
                        edgecolor=colors[0],  # red
                        lw=3,
                    )
                )
            elif home_goal == away_goal:  # draw
                axes.add_patch(
                    Rectangle(
                        (rows - row - 1, cols - col - 1),
                        1,
                        1,
                        fill=False,
                        edgecolor=colors[1],  # black
                        lw=3,
                    )
                )
            else:  # away win
                axes.add_patch(
                    Rectangle(
                        (rows - row - 1, cols - col - 1),
                        1,
                        1,
                        fill=False,
                        edgecolor=colors[2],  # blue
                        lw=3,
                    )
                )
    axes.invert_yaxis()
    # calculate the sum of the 3 outcomes for the legend entry from the crosstab (we can also do it from the filtered df with a condition)
    home_perc_sum = np.sum(
        goals_perc.values[np.tril_indices(goals_perc.shape[0], k=-1)]
    )
    draw_perc_sum = np.trace(goals_perc.values)
    away_perc_sum = np.sum(goals_perc.values[np.triu_indices(goals_perc.shape[0], k=1)])
    leg = plt.legend(
        title="Ratio of match outcomes",
        title_fontsize=8,
        fontsize=7,
        labelcolor=colors,
        labels=[
            f"Home Win {home_perc_sum.round(2)}%",
            f"Draw {draw_perc_sum.round(2)}%",
            f"Away Win {away_perc_sum.round(2)}%",
        ],
        loc="lower right",
        bbox_to_anchor=(0, 0.92),
        frameon=False,
    )
    for ii, entries in enumerate(leg.legend_handles):
        entries.set_color(colors[ii])
