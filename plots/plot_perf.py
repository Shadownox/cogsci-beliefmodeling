import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import sys

if __name__ == "__main__":
    
    filename = None
    
    # check if file is given
    if len(sys.argv) == 1:
        filename = 'results/2021-01-27-with_portfolio.csv'
    else:
        filename = sys.argv[1]

    acc_models = ['FOL', 'MN-FOL', 'MN-PHM', 'MN-mReasoner', 'PHM', 'Random', 'SS-FOL', 'SS-PHM',
     'SS-mReasoner', 'UserMedian', 'mReasoner', 'ver-Portfolio', 'ver-Portfolio-FOL', 'ver-Portfolio-PHM',
     'ver-Portfolio-mReasoner']
    rat_models = ['FOL', 'MN-FOL', 'MN-PHM', 'MN-mReasoner', 'PHM', 'Random', 'SS-FOL', 'SS-PHM',
     'SS-mReasoner', 'UserMedian', 'mReasoner', 'rat-Portfolio',
     'rat-Portfolio-FOL', 'rat-Portfolio-PHM', 'rat-Portfolio-mReasoner']

    def get_hue(elem):
        if elem.startswith("MN"):
            return "with MN"
        elif elem.startswith("SS"):
            return "with SS"
        elif elem.startswith("indiv-"):
            return "indiv. belief"
        else:
            return "Model only"

    data_df = pd.read_csv(filename)
    acc_df = data_df[data_df['model'].isin(acc_models)].copy()
    rat_df = data_df[data_df['model'].isin(rat_models)].copy()

    acc_df["model"] = acc_df["model"].apply(lambda y: y.replace("ver-", "").replace("UserMedian", "PersonMean").replace("Portfolio-", "indiv-"))
    rat_df["model"] = rat_df["model"].apply(lambda y: y.replace("rat-", "").replace("UserMedian", "PersonMean").replace("Portfolio-", "indiv-"))

    acc_df["Belief"] = acc_df["model"].apply(get_hue)
    rat_df["Belief"] = rat_df["model"].apply(get_hue)

    acc_df["HueModelName"] = acc_df["model"].apply(lambda y: y[y.find("-")+1:])
    rat_df["HueModelName"] = rat_df["model"].apply(lambda y: y[y.find("-")+1:])


    colorcodes = {
        'Random': 2,
        'Portfolio': 2,
        'PersonMean': 2,
        'indiv-PHM' : 2,
        'indiv-FOL' : 2,
        'indiv-mReasoner' : 2,
        'FOL': 0,
        'PHM': 0,
        'mReasoner': 0
    }

    # overall performance
    f = plt.figure(figsize=(9, 4))
    sns.set(palette="colorblind")
    sns.set_style("whitegrid", {'axes.grid' : True})

    order = acc_df.groupby('model', as_index=False)['score_response'].agg('mean').sort_values('score_response')['model']
    color_palette = ['C{}'.format(colorcodes[x] if x in colorcodes else 1) for x in order]
    ax = sns.barplot(x="model", y="score_response", data=acc_df, palette=color_palette, order=order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(ylim=(0, 0.8))

    ax.set_xlabel("")
    ax.set_ylabel("Predictive Accuracy")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=15, label='No Belief'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', markersize=15, label='Belief Models'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=15, label='Baselines'),
    ]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc='center',
        ncol=3,
        borderaxespad=0.,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig("belief_performance.pdf")
    plt.show()

    # rating performance
    f = plt.figure(figsize=(9, 4))
    sns.set(palette="colorblind")
    sns.set_style("whitegrid", {'axes.grid' : True})

    order = rat_df.groupby('model', as_index=False)['score_rating'].agg('mean').sort_values('score_rating')['model']
    color_palette = ['C{}'.format(colorcodes[x] if x in colorcodes else 1) for x in order]
    ax = sns.barplot(x="model", y="score_rating", data=rat_df, palette=color_palette, order=order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=15, label='No Belief'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', markersize=15, label='Belief Models'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=15, label='Baselines'),
    ]

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc='center',
        ncol=3,
        borderaxespad=0.,
        frameon=False
    )

    ax.set_xlabel("")
    ax.set_ylabel("Absolute difference")
    plt.tight_layout()
    plt.savefig("rating_performance.pdf")
    plt.show()

