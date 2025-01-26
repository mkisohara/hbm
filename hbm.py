import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be", category=ResourceWarning)

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os
import seaborn as sns
from matplotlib import _mathtext as mathtext
from concurrent.futures import ProcessPoolExecutor
mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
plt.rcParams.update({'mathtext.default': 'default',
                     'mathtext.fontset': 'stix',
                     'font.family': 'Arial',
                     })
user_home_path = os.path.expanduser("~")
resultdir = "~" #Please define as appropriate.
os.makedirs(resultdir, exist_ok=True)

# --- 1. Preparing data ---
np.random.seed(42)
n_data_samples = 170
xs = 0
xe = 2000

def f(x, beta1=0.15, beta2=5.01, beta3=0.96):
    return beta1 * x + beta2 * np.sqrt(np.abs(x)) + beta3

def generate_data_gamma_x(xs, xe, n_samples, seed, prob=0.95):
    np.random.seed(seed)  # Set fixed seed
    x_values = []
    y_values = []
    shape_param = 2.0
    scale_param = xe / (5*shape_param)
    while len(x_values) < n_samples:
        x = gamma.rvs(a=shape_param, scale=scale_param)
        if x > xe or x < xs:
            continue
        if np.random.rand() <= prob:
            y = np.random.uniform(-f(x), f(x))
        else:
            if np.random.rand() < 0.5:
                y = np.random.uniform(f(x), f(x) + 100)
            else:
                y = np.random.uniform(-f(x) - 100, -f(x))
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

simulation_array = np.arange(xs, xe, 10) 

def h_func_chung(m, beta1=0.17, beta2=5.0, beta3=1.08):
    return beta1 * m + beta2 * np.sqrt(m) + beta3
hs_cs = h_func_chung(simulation_array)
def h_func_nina(m, beta1=0.15, beta2=5.01, beta3=0.96):
    return beta1 * m + beta2 * np.sqrt(m) + beta3
hs_np = h_func_nina(simulation_array)

# --- Set plot ---
sns.set(style='darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes = axes.flatten()
seeds = [40, 50, 60, 70]
labels = ['A', 'B', 'C', 'D']

for idx, seed in enumerate(seeds):
    ax = axes[idx]
    
    # Generating data
    x_data, y_data = generate_data_gamma_x(xs, xe, n_data_samples, seed, prob=0.95)


    # Structure of PyMC and Sampling
    with pm.Model() as model:
        beta1 = pm.Normal('α', mu=0.17, sigma=10)
        beta2 = pm.Normal('β', mu=5.01, sigma=10)
        beta3 = pm.Normal('γ', mu=1.08, sigma=10)

        sigma_vals = (beta1 * x_data + beta2 * np.sqrt(np.abs(x_data)) + beta3) / 1.96
        y_obs = pm.Normal('y_obs', mu=0, sigma=sigma_vals, observed=y_data)

        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=False)

    # Post-event sample extraction
    posterior = trace.posterior
    posterior_stack = posterior.stack(sample=("chain", "draw"))
    beta1_samples = posterior_stack['α'].values
    beta2_samples = posterior_stack['β'].values
    beta3_samples = posterior_stack['γ'].values
    n_samples = beta1_samples.shape[0]

    grid_x = np.linspace(xs, xe, 100)
    n_grid = len(grid_x)

    # Calculating g(x) on a grid
    g_samples = np.zeros((n_samples, n_grid))
    for j in range(n_samples):
        g_samples[j, :] = beta1_samples[j] * grid_x + beta2_samples[j] * np.sqrt(np.abs(grid_x)) + beta3_samples[j]

    upper_samples = g_samples        
    lower_samples = -g_samples       

    upper_mean = upper_samples.mean(axis=0)
    lower_mean = lower_samples.mean(axis=0)

    upper_ci_lower = np.percentile(upper_samples, 2.5, axis=0)
    upper_ci_upper = np.percentile(upper_samples, 97.5, axis=0)
    lower_ci_lower = np.percentile(lower_samples, 2.5, axis=0)
    lower_ci_upper = np.percentile(lower_samples, 97.5, axis=0)

    ax.axhline(y=np.mean(y_data), color='gray', linestyle='--', linewidth=1)
    ax.scatter(x_data, y_data, color='blue', edgecolor='k', s=12, alpha=0.6)
    ax.plot(grid_x, upper_mean, color='red', label='Upper LoA Mean with HBM')
    ax.plot(grid_x, lower_mean, color='blue', label='Lower LoA Mean with HBM')
    ax.fill_between(grid_x, upper_ci_lower, upper_ci_upper, color='red', alpha=0.2, label='95% CI for upper LoA')
    ax.fill_between(grid_x, lower_ci_lower, lower_ci_upper, color='blue', alpha=0.2, label='95% CI for lower LoA')
    ax.plot(simulation_array, hs_np, ':', color='magenta', label='Haag et al')
    ax.plot(simulation_array, -hs_np, ':', color='magenta')
    ax.plot(simulation_array, hs_cs, ':', color='darkcyan', label='Chung et al')
    ax.plot(simulation_array, -hs_cs, ':', color='darkcyan')
    ax.set_xlim(0, xe)
    ax.set_ylim(-1200, 1200)

    # Axis label settings (outer circumference only)
    if idx in [2, 3]:
        ax.set_xlabel('Mean')
    if idx in [0, 2]:
        ax.set_ylabel('Difference')

    # Subplot labels
    ax.text(0.05, 0.85, f'{labels[idx]}', style='italic', 
            fontsize=12, backgroundcolor="white", transform=ax.transAxes)

# Only the outer scale labels are displayed.
for ax in axes:
    ax.label_outer()


# Create a common legend
handles, labels_legend = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels_legend, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower center',
           fontsize=12, ncol=3, bbox_to_anchor=(0.5, -0.01))
# Adjust the margins between subplots
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.2, top=0.9, wspace=0.1, hspace=0.03)
# Save the plot
save_path = os.path.join(resultdir, f'HBM_ci_2x2_plots.tif')
plt.savefig(save_path, dpi=100, bbox_inches='tight')
