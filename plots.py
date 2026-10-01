import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_BAR_COLORS = ["#2b6777", "#52ab98", "#c8d8e4", "#f2a65a"]


def print_header(title, width=50):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def corr_heatmap(df, columns):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df[columns].corr(), annot=True, cmap="coolwarm",
                linewidths=0.5, annot_kws={"size": 8})
    plt.title("Correlation Heatmap", fontsize=10, fontweight="bold", pad=8)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.show()

#Histogram + KDE per column, x-axis clipped at the 99th percentile
def plot_distributions(df, columns, ncols=2):
    
    print("Columns distribution before handling")
    nrows = int(np.ceil(len(columns) / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=100, shrink=0.8,
                     color="#2b6777", edgecolor="white")
        ax.set_xlim(0, df[col].quantile(0.99))
        ax.set_title(f"Distribution of {col}", fontsize=9, fontweight="bold")
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("Frequency", fontsize=7)

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.tight_layout(pad=10.0)
    plt.show()

#Bar chart of a DataFrame with 'Feature' and 'Relative_Importance_%' columns
def plot_top_features(top_df, title):
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Relative_Importance_%", y="Feature", data=top_df, palette="viridis")
    plt.title(title)
    plt.xlabel("Relative Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

#results maps model name -> {'accuracy', 'train_time', 'test_time'}.
def plot_classifier_summary(results, output_path="classifier_summary.png"):
    
    models = list(results)
    x = np.arange(len(models))
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    train_times = [results[m]["train_time"] for m in models]
    test_times = [results[m]["test_time"] for m in models]

    # (title, y-label, values, label format, label offset, y-limit)
    panels = [
        ("Classification Accuracy (%)", "Accuracy (%)", accuracies, "{:.2f}%", 1.5, (0, 110)),
        ("Total Training Time (s)", "Time (seconds)", train_times, "{:.2f}s",
         0.02 * max(train_times), None),
        ("Total Test Time (s)", "Time (seconds)", test_times, "{:.4f}s",
         0.02 * max(test_times), None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Classifier Performance Summary", fontsize=14, fontweight="bold", y=1.02)

    for ax, (title, ylabel, values, fmt, offset, ylim) in zip(axes, panels):
        bars = ax.bar(x, values, width=0.55, color=_BAR_COLORS, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    fmt.format(value), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {output_path}")
