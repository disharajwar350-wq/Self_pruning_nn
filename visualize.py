import json
import numpy as np
import matplotlib.pyplot as plt


LAMS   = [1e-5, 1e-4, 1e-3]
COLORS = ['#2196F3', '#FF9800', '#F44336']
CKPT   = './checkpoints'


def load(lam, key):
    with open(f'{CKPT}/lambda_{lam}/history.json') as f:
        return json.load(f)[key]


def plot_gate_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Gate Value Distributions', fontsize=13, fontweight='bold')

    for ax, lam, c in zip(axes, LAMS, COLORS):
        gates    = np.array(load(lam, 'gates'))
        sparsity = 100 * (gates < 0.01).sum() / len(gates)
        ax.hist(gates, bins=60, color=c, alpha=0.8, edgecolor='white')
        ax.axvline(0.01, color='black', linestyle='--', linewidth=1, label='threshold')
        ax.set_title(f'λ={lam}  |  {sparsity:.1f}% pruned', fontweight='bold')
        ax.set_xlabel('gate value')
        ax.set_ylabel('count' if ax is axes[0] else '')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('./plots/gate_distribution.png', bbox_inches='tight')
    plt.close()
    print('saved gate_distribution.png')


def plot_accuracy():
    plt.figure(figsize=(8, 5))
    for lam, c in zip(LAMS, COLORS):
        acc = load(lam, 'acc')
        plt.plot(range(1, len(acc)+1), acc, color=c, linewidth=2, label=f'λ={lam}')
    plt.title('Test Accuracy over Epochs', fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/accuracy.png', bbox_inches='tight')
    plt.close()
    print('saved accuracy.png')


def plot_sparsity():
    plt.figure(figsize=(8, 5))
    for lam, c in zip(LAMS, COLORS):
        sp = load(lam, 'sparsity')
        plt.plot(range(1, len(sp)+1), sp, color=c, linewidth=2, label=f'λ={lam}')
    plt.title('Sparsity Growth over Epochs', fontweight='bold')
    plt.xlabel('epoch')
    plt.ylabel('sparsity (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/sparsity.png', bbox_inches='tight')
    plt.close()
    print('saved sparsity.png')


def plot_tradeoff(results):
    plt.figure(figsize=(6, 5))
    for r, c in zip(results, COLORS):
        plt.scatter(r['final_sparsity'], r['final_acc'],
                    s=150, color=c, zorder=5, edgecolors='white')
        plt.annotate(f"λ={r['lam']}",
                     (r['final_sparsity'], r['final_acc']),
                     xytext=(6, 3), textcoords='offset points',
                     fontsize=9, color=c, fontweight='bold')
    plt.xlabel('sparsity (%)')
    plt.ylabel('accuracy (%)')
    plt.title('Accuracy vs Sparsity', fontweight='bold')
    plt.tight_layout()
    plt.savefig('./plots/tradeoff.png', bbox_inches='tight')
    plt.close()
    print('saved tradeoff.png')


if __name__ == '__main__':
    plot_gate_distribution()
    plot_accuracy()
    plot_sparsity()