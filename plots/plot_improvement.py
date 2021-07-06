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
        filename = 'results/2021-01-26_models_only.csv'
    else:
        filename = sys.argv[1]

    # Load data
    df = pd.read_csv(filename)

    # Remove models
    BLACKLIST = [
        'Random',
        'rat-Portfolio-FOL',
        'rat-Portfolio-PHM',
        'rat-Portfolio-mReasoner',
        'ver-Portfolio',
        'ver-Portfolio-FOL',
        'ver-Portfolio-PHM',
        'ver-Portfolio-mReasoner',
        'rat-Portfolio',
        'UserMedian'
    ]
    
    df = df[~df['model'].isin(BLACKLIST)]

    # Add columns
    def get_model_type(x):
        if 'MN-' in x:
            return 'MisinterpretedNecessity'
        if 'SS-' in x:
            return 'SelectiveScrutiny'
        else:
            return 'NoBelief'
    df['mtype'] = df['model'].apply(get_model_type)
    df['mbase'] = df['model'].apply(lambda x: x.split('-')[1] if '-' in x else x)

    # Aggregate data
    df_agg = df.groupby(['mtype', 'mbase', 'id'], as_index=False)['score_response'].agg('mean')

    # Plot
    sns.set(style='whitegrid', palette='colorblind')
    plt.figure(figsize=(7, 3))

    hue_order = ['NoBelief', 'SelectiveScrutiny', 'MisinterpretedNecessity']
    bars = sns.barplot(x='mbase', y='score_response', hue='mtype', data=df_agg, hue_order=hue_order)

    # Add text
    yoffset = -0.05
    for bar in [x for x in bars.get_children() if isinstance(x, matplotlib.patches.Rectangle)]:
        if bar.get_width() > 0.5:
            continue

        xpos = bar.get_x() + bar.get_width() / 2
        ypos = bar.get_height() + yoffset
        plt.text(x=xpos, y=ypos, s=np.round(bar.get_height(), 2), ha='center', va='top', color='white')

    plt.xlabel('')
    plt.ylabel('Predictive Accuracy')
    plt.yticks(np.arange(0, 0.71, 0.1))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=15, label='No Belief'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', markersize=15, label='Selective Scrutiny'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=15, label='Misinterpreted Necessity'),
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
    plt.savefig('plot_improvement.pdf')
    plt.show()
