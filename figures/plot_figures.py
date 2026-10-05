from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(figure, name):
    figure.savefig(ROOT / (name + '.pdf'), metadata={'Author': '', 'CreationDate': None, 'ModDate': None}, bbox_inches='tight')
    figure.savefig(ROOT / (name + '.png'), dpi=1000, bbox_inches='tight')
    figure.savefig(ROOT / (name + '.svg'), metadata={'Date': None}, bbox_inches='tight')
    plt.close(figure)


def ranks():
    data = pd.read_csv(ROOT / 'INTELLIGENCE_Figure_1_data.csv')
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1.15, 1]})
    axes[0].scatter(data.rank_qnw, data.rank_hlo, s=24, color='#24445f', alpha=.8)
    axes[0].plot([1, 66], [1, 66], linestyle='--', color='.5', linewidth=1)
    for row in data[data.absolute_rank_displacement >= 20].itertuples():
        axes[0].annotate(row.iso3, (row.rank_qnw, row.rank_hlo), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='Published QNW rank', ylabel='Aggregate HLO rank', xlim=(68, 0), ylim=(68, 0))
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title('A  Reported point-score ranks', loc='left', fontsize=10)
    bins = np.arange(-.5, 36.5, 2)
    axes[1].hist(data.absolute_rank_displacement, bins=bins, color='#24445f', edgecolor='white')
    axes[1].axvline(data.absolute_rank_displacement.mean(), color='#a34722', linestyle='--', label=f'Mean {data.absolute_rank_displacement.mean():.2f}')
    axes[1].set(xlabel='Absolute rank displacement (places)', ylabel='Number of economies', xlim=(-1, 36))
    axes[1].set_title('B  Distribution across 66 economies', loc='left', fontsize=10)
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    save(figure, 'INTELLIGENCE_Figure_1')


def performance():
    data = pd.read_csv(ROOT / 'INTELLIGENCE_Figure_2_data.csv')
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    styles = [('ridge', '#24445f', 'o', 'Ridge (primary)'), ('ols', '#a34722', 's', 'OLS'), ('lasso', '#667b45', '^', 'Lasso')]
    for axis, direction, title in zip(axes, ['HLO_to_P', 'P_to_HLO'], ['A  HLO predicts period-specific P', 'B  Period-specific P predicts HLO']):
        group = data[data.direction == direction]
        for offset, (estimator, color, marker, label) in zip([-.15, 0, .15], styles):
            selected = group[group.estimator == estimator].set_index('model_id').loc[['M1', 'M2', 'M3']]
            axis.scatter(np.arange(3) + offset, selected.normalized_mse, marker=marker, color=color, label=label, s=45)
        axis.axhline(1, color='.5', linestyle='--', linewidth=1, label='Mean-only reference')
        axis.set(xticks=np.arange(3), xticklabels=['Other indicator', 'Seven context\nfeatures', 'Indicator and\ncontext'], ylim=(0, 1.12))
        axis.set_title(title, loc='left', fontsize=10)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Country-weighted MSE / direction-specific reference MSE')
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    figure.tight_layout()
    save(figure, 'INTELLIGENCE_Figure_2')


if __name__ == '__main__':
    ranks()
    performance()
