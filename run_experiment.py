import os
from visualize import plot_gate_distribution, plot_accuracy, plot_sparsity, plot_tradeoff
from train import train

LAMBDAS = [1e-5, 1e-4, 1e-3]

os.makedirs('./plots', exist_ok=True)

results = []
for lam in LAMBDAS:
    h = train(lam=lam, epochs=40)
    results.append({
        'lam':            lam,
        'final_acc':      h['final_acc'],
        'final_sparsity': h['final_sparsity'],
    })

plot_gate_distribution()
plot_accuracy()
plot_sparsity()
plot_tradeoff(results)

print('\nresults')
print(f"{'lambda':<10} {'accuracy':>10} {'sparsity':>10}")
print('-' * 32)
for r in results:
    print(f"{r['lam']:<10} {r['final_acc']:>9.2f}% {r['final_sparsity']:>9.1f}%")

print('\nmarkdown table')
print('| Lambda | Test Accuracy | Sparsity (%) |')
for r in results:
    print(f"| {r['lam']} | {r['final_acc']:.2f}% | {r['final_sparsity']:.1f}% |")