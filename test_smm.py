import sys, json
sys.path.insert(0, '.')
from residual_model import ResidualAnomalyModel
from semi_markov import SemiMarkovFitter
import numpy as np

print('Running NVDA / QQQ / SMH analysis...')
model = ResidualAnomalyModel()
analysis = model.analyze(asset_ticker='NVDA', market_ticker='QQQ', sector_ticker='SMH')

if analysis.get('status') != 'SUCCESS':
    print('Analysis failed:', analysis.get('error'))
    sys.exit(1)

history = analysis.get('history', [])
print(f'History length: {len(history)} trading days')

z_series = np.array([float(h.get('z_1d', 0.0)) for h in history])

smf = SemiMarkovFitter(threshold=1.8)
result = smf.fit(z_series)

print(f'Fit quality    : {result["fit_quality"]}')
print(f'Transitions    : {result["n_transitions"]}')
print(f'Current regime : {result["current_state_name"]} ({result["days_in_regime"]} days)')
print(f'Exit prob 1d   : {result["exit_prob_1d"]:.1%}')
print(f'Exit prob 3d   : {result["exit_prob_3d"]:.1%}')
print(f'Exit prob 5d   : {result["exit_prob_5d"]:.1%}')
print(f'Mean resid life: {result["mean_residual_life"]:.1f} days')

for s, fd in result['sojourn_fits'].items():
    print(f'  State {s}: {json.dumps(fd)}')

P = result['transition_matrix']
labels = ['UNDER','NEUT ','OVER ']
print('Transition matrix:')
for i, row in enumerate(P):
    print(f'  {labels[i]}  {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}')
