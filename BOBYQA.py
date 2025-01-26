import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be", category=ResourceWarning)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from matplotlib import _mathtext as mathtext
import os
import pybobyqa
import seaborn as sns
import concurrent.futures 
from concurrent.futures import ProcessPoolExecutor

mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants

plt.rcParams.update({'mathtext.default': 'default',
                     'mathtext.fontset': 'stix',
                     'font.family': 'Arial',
                     })

user_home_path = os.path.expanduser("~")
resultdir = os.path.join(user_home_path, 'Dropbox', 'Medicine', '研究', 'Cardiovascular', 'P6_9')
os.makedirs(resultdir, exist_ok=True)

def f(x, beta1=0.15, beta2=5.01, beta3=0.96):
    return beta1 * x + beta2 * np.sqrt(np.abs(x)) + beta3

def generate_data_gamma_x(xs, xe, n_samples, seed, prob=0.95):
    np.random.seed(seed)  # Set fixed seed
    x_values = []
    y_values = []
    shape_param = 2.0
    scale_param = xe / (5*shape_param)
    for _ in range(n_samples):
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

global_x1 = None
global_x2 = None

def coverage_loss(params, x1, x2, coverage_target=0.95, w=100.0):
    beta1, beta2, beta3 = params
    mean_values = (x1 + x2) / 2.0
    h_vals = beta1 * mean_values + beta2 * np.sqrt(mean_values) + beta3
    param_size_loss = beta1 + beta2 + beta3
    d = x1 - x2
    inside_loa = (d >= -h_vals) & (d <= h_vals)
    actual_coverage = np.mean(inside_loa)
    coverage_error = abs(coverage_target - actual_coverage)
    loss_value = param_size_loss + w * coverage_error
    return loss_value

def init_worker(x1, x2):
    global global_x1, global_x2
    global_x1, global_x2 = x1, x2

def fit_single_seed1(seed, x1, x2):
    import pybobyqa
    np.random.seed(seed)
    # Fixed initial values for repeatability (change as needed)
    #x0 = np.random.uniform(0, 10, size=3)
    x0 = [0.17, 5.0, 1.08]  # Chung's regression equation
    lower_bounds = [0, 0, 0]
    upper_bounds = [100, 100, 100]
    beta_history = []
    loss_history = []

    def coverage_loss_with_history(params, x1_, x2_, coverage_target=0.95, w=100.0):
        beta_history.append(params.copy())
        lv = coverage_loss(params, x1_, x2_, coverage_target, w)
        loss_history.append(lv)
        return lv

    res = pybobyqa.solve(
        coverage_loss_with_history,
        x0=x0,
        args=(x1, x2),
        bounds=(lower_bounds, upper_bounds),
        rhobeg=2,
        rhoend=1e-16,
        maxfun=2000
    )
    return beta_history, loss_history, res.x

def fit_single_seed_only1(seed):
    return fit_single_seed1(seed, global_x1, global_x2)


def fit_single_seed(seed, x1, x2):
    np.random.seed(seed)
    x0 = np.random.uniform(0, 10, size=3)
    lower_bounds = [0, 0, 0]
    upper_bounds = [100, 100, 100]
    beta_history = []
    loss_history = []

    def coverage_loss_with_history(params, x1_, x2_, coverage_target=0.95, w=100.0):
        beta_history.append(params.copy())
        lv = coverage_loss(params, x1_, x2_, coverage_target, w)
        loss_history.append(lv)
        return lv

    res = pybobyqa.solve(
        coverage_loss_with_history,
        x0=x0,
        args=(x1, x2),
        bounds=(lower_bounds, upper_bounds),
        rhobeg=2,
        rhoend=1e-16,
        maxfun=2000
    )
    return beta_history, loss_history, res.x

def EstimateBeta_BOBYQA_2x2plots(xs, xe, n_samples, sample_seeds, num_bobyqa_seeds):
    """
    Performs data generation and optimization for 4 specified seeds,
    Plot the results on a 2x2 subplot.
    """
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.flatten()
    simulation_array = np.arange(xs, xe, 10)  # Specify x-axis range and step

    labels = ['A', 'B', 'C', 'D']

    for ax, sample_seed, label in zip(axes, sample_seeds, labels):
        all_beta_histories1 = []
        all_loss_histories1 = []
        final_betas1 = []

        # Generating data
        x_data_gamma, y_data_gamma = generate_data_gamma_x(xs, xe, n_samples, sample_seed, prob=0.95)
        x1 = x_data_gamma + y_data_gamma/2
        x2 = x_data_gamma - y_data_gamma/2
        simulation_array = np.arange(xs, xe, 10)

        # BOBYQA Fixed Param
        with ProcessPoolExecutor(initializer=init_worker, initargs=(x1, x2)) as executor:
            results1 = list(executor.map(fit_single_seed_only1, range(num_bobyqa_seeds)))

        for beta_history, loss_history, final_beta in results1:
            all_beta_histories1.append(beta_history)
            all_loss_histories1.append(loss_history)
            final_betas1.append(final_beta)
        beta11 = [b[0] for b in final_betas1][-1]
        beta21 = [b[1] for b in final_betas1][-1]
        beta31 = [b[2] for b in final_betas1][-1]
        localoptiima = beta11 * simulation_array + beta21 * np.sqrt(simulation_array) + beta31

        # Parallel Processing of Optimization
        bobyqa_seeds = list(range(num_bobyqa_seeds))
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(fit_single_seed, bobyqa_seed, x1, x2) for bobyqa_seed in bobyqa_seeds]
            for future in concurrent.futures.as_completed(futures):
                _, _, final_beta = future.result()
                beta1, beta2, beta3 = final_beta
                # Calculate LoA values using simulation_array
                loa_values = beta1 * simulation_array + beta2 * np.sqrt(simulation_array) + beta3
                results.append(loa_values)

        results_array = np.array(results)
        mean_loa_values = np.mean(results_array, axis=0)
        std_loa_values = np.std(results_array, axis=0)
        ci_upper = mean_loa_values + 1.96 * std_loa_values  # Upper 95% confidence interval
        ci_lower = mean_loa_values - 1.96 * std_loa_values  # Lower 95% confidence interval

        def h_func_chung(m, beta1=0.17, beta2=5.0, beta3=1.08):
            return beta1 * m + beta2 * np.sqrt(m) + beta3
        hs_cs = h_func_chung(simulation_array)

        def h_func_nina(m, beta1=0.15, beta2=5.01, beta3=0.96):
            return beta1 * m + beta2 * np.sqrt(m) + beta3
        hs_np = h_func_nina(simulation_array)

        # Plotting
        for i, loa_values in enumerate(results):
            if i == 0:
                ax.plot(simulation_array, localoptiima, ':', color='black', alpha=1, label='Local optima')
                ax.plot(simulation_array, -localoptiima, ':', color='black')
                ax.plot(simulation_array, hs_np, ':', color='magenta', label='Haag et al')
                ax.plot(simulation_array, -hs_np, ':', color='magenta')
                ax.plot(simulation_array, hs_cs, ':', color='darkcyan', label='Chung et al')
                ax.plot(simulation_array, -hs_cs, ':', color='darkcyan')
            else:
                ax.plot(simulation_array, localoptiima, ':', color='black', alpha=1)
                ax.plot(simulation_array, -localoptiima, ':', color='black')
                ax.plot(simulation_array, hs_np, ':', color='magenta', alpha=0.1)
                ax.plot(simulation_array, -hs_np, ':', color='magenta', alpha=0.1)
                ax.plot(simulation_array, hs_cs, ':', color='darkcyan', alpha=0.1)
                ax.plot(simulation_array, -hs_cs, ':', color='darkcyan', alpha=0.1)
        
        mean_values = (x1 + x2) / 2.0
        d_values = x1 - x2
        ax.fill_between(simulation_array, ci_lower, ci_upper, color='purple', alpha=0.1, label='95% CI')
        ax.fill_between(simulation_array, -ci_lower, -ci_upper, color='purple', alpha=0.1, label='95% CI')
        ax.axhline(y=np.mean(d_values), color='gray', linestyle='--', linewidth=1, label='Mean difference')
        ax.scatter(mean_values, d_values, color='blue', edgecolor='k', s=12, alpha=0.6)
        ax.set_xlim(0, xe)
        ax.set_ylim(-1200, 1200)
        if label == 'C' or label == "D":
            ax.set_xlabel('Mean')
        if label == 'A' or label == "C":
            ax.set_ylabel('Difference')
        ax.text(0.05, 0.85, f'{label}', style='italic', math_fontfamily='cm', fontsize=12,
                backgroundcolor="white", transform=ax.transAxes) 

    for ax in axes:
        ax.label_outer()

    handles, labels_legend = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               fontsize=12, ncol=3, bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.2, top=0.9, wspace=0.1, hspace=0.03)
    save_path = os.path.join(resultdir, f'2x2_plots_different_data_seeds_{"_".join(map(str, sample_seeds))}.tif')
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    seeds = [40, 50, 60, 70] 
    num_bobyqa_seeds = 2000
    xs = 0
    xe = 2000
    n_samples = 170
    EstimateBeta_BOBYQA_2x2plots(xs, xe, n_samples, seeds, num_bobyqa_seeds)