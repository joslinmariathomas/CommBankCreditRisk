import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_target_by_feature(df:pd.DataFrame, contract_col: str):


    target_col = "TARGET"

    target_totals = df[target_col].value_counts().sort_values(ascending=False)
    ordered_targets = target_totals.index.tolist()

    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        data=df,
        x=target_col,
        hue=contract_col,
        order=ordered_targets,
        ax=ax
    )

    # Styling (similar to original)
    ax.set_xlabel(str(target_col))
    ax.set_ylabel("Count")
    title = f"{target_col} by {contract_col}"
    ax.set_title(title)
    ax.legend(title=str(contract_col))
    fig.tight_layout()



    df.groupby([target_col, contract_col]).size().unstack(fill_value=0).reindex(ordered_targets)

    plt.show()


def plot_kde_by_feature(df:pd.DataFrame,feature:str,
                        ):
    """
    Plot overlaid KDE curves of x_col by hue_col from application_train using seaborn.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df,
        x=feature,
        hue='TARGET',
        common_norm=False,
        bw_method='scott',
        fill=False,
        linewidth=2
    )

    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()