import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import argparse
import sys

from matplotlib.colors import LogNorm

sys.path.append('experiments/exchange_rates')
from evaluate_exchange_rates import X_values, N_values, inflect_option

model_key_to_name = {
    # ------------------------------
    # OpenAI Models
    # ------------------------------
    "gpt-35-turbo": "GPT 3.5 Turbo",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
    "o1-mini": "o1 Mini",
    "o1": "o1",

    # ------------------------------
    # Anthropic Models
    # ------------------------------
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "claude-3-haiku": "Claude 3 Haiku",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "claude-2.1": "Claude 2.1",
    "claude-2": "Claude 2",
    "claude-instant-1.2": "Claude Instant 1.2",

    # ------------------------------
    # Google Models
    # ------------------------------
    "gemini-pro": "Gemini Pro",
    "gemini-15-flash": "Gemini 1.5 Flash",
    "gemini-20-flash-exp": "Gemini 2.0 Flash Exp",
    "gemini-15-pro": "Gemini 1.5 Pro",

    # ------------------------------
    # XAI Models
    # ------------------------------
    "grok-2-1212": "Grok 2 1212",

    # ------------------------------
    # Custom Models
    # ------------------------------
    "llama-31-8b-instruct-citizen-assembly":  "Llama 3.1 8B Instruct Citizen Assembly",
    "llama-31-8b-instruct-citizen-assembly2": "Llama 3.1 8B Instruct Citizen Assembly 2",
    "llama-31-8b-instruct-citizen-assembly3": "Llama 3.1 8B Instruct Citizen Assembly 3",
    "llama-31-8b-instruct-citizen-assembly4": "Llama 3.1 8B Instruct Citizen Assembly 4",

    # ------------------------------
    # Meta Llama
    # ------------------------------
    "llama-2-7b":                 "Llama 2 7B",
    "llama-2-7b-instruct":                 "Llama 2 7B",
    "llama-2-13b":                 "Llama 2 13B",
    "llama-2-13b-instruct":                 "Llama 2 13B",
    "llama-2-70b":                 "Llama 2 70B",
    "llama-2-70b-instruct":                 "Llama 2 70B",
    "llama-32-1b":                 "Llama 3.2 1B",
    "llama-32-1b-instruct":        "Llama 3.2 1B",
    "llama-32-3b":                 "Llama 3.2 3B",
    "llama-32-3b-instruct":        "Llama 3.2 3B",
    "llama-31-8b":                 "Llama 3.1 8B",
    "llama-31-8b-instruct":        "Llama 3.1 8B",
    "llama-31-70b":                "Llama 3.1 70B",
    "llama-31-70b-instruct":       "Llama 3.1 70B",
    "llama-33-70b-instruct":       "Llama 3.3 70B",
    "llama-31-405b-fp8":           "Llama 3.1 405B",
    "llama-31-405b-instruct-fp8":  "Llama 3.1 405B",

    # ------------------------------
    # Qwen 1.5
    # ------------------------------
    "qwen15-05b":           "Qwen1.5 0.5B",
    "qwen15-05b-instruct":  "Qwen1.5 0.5B Chat",
    "qwen15-18b":           "Qwen1.5 1.8B",
    "qwen15-18b-instruct":  "Qwen1.5 1.8B Chat",
    "qwen15-4b":            "Qwen1.5 4B",
    "qwen15-4b-instruct":   "Qwen1.5 4B Chat",
    "qwen15-7b":            "Qwen1.5 7B",
    "qwen15-7b-instruct":   "Qwen1.5 7B Chat",
    "qwen15-14b":           "Qwen1.5 14B",
    "qwen15-14b-instruct":  "Qwen1.5 14B Chat",
    "qwen15-32b":           "Qwen1.5 32B",
    "qwen15-32b-instruct":  "Qwen1.5 32B Chat",
    "qwen15-72b":           "Qwen1.5 72B",
    "qwen15-72b-instruct":  "Qwen1.5 72B Chat",
    "qwen15-110b":          "Qwen1.5 110B",
    "qwen15-110b-instruct": "Qwen1.5 110B Chat",

    # ------------------------------
    # Qwen 2.5
    # ------------------------------
    "qwen25-05b":           "Qwen2.5 0.5B",
    "qwen25-05b-instruct":  "Qwen2.5 0.5B",
    "qwen25-15b":           "Qwen2.5 1.5B",
    "qwen25-15b-instruct":  "Qwen2.5 1.5B",
    "qwen25-3b":            "Qwen2.5 3B",
    "qwen25-3b-instruct":   "Qwen2.5 3B",
    "qwen25-7b":            "Qwen2.5 7B",
    "qwen25-7b-instruct":   "Qwen2.5 7B",
    "qwen25-14b":           "Qwen2.5 14B",
    "qwen25-14b-instruct":  "Qwen2.5 14B",
    "qwen25-32b":           "Qwen2.5 32B",
    "qwen25-32b-instruct":  "Qwen2.5 32B",
    "qwq-32b-preview":      "QwQ 32B Preview",
    "qwen25-72b":           "Qwen2.5 72B",
    "qwen25-72b-instruct":  "Qwen2.5 72B",

    # ------------------------------
    # Google Gemma
    # ------------------------------
    "gemma-2-2b":    "Gemma 2 2B",
    "gemma-2-2b-it": "Gemma 2 2B IT",
    "gemma-2-9b":    "Gemma 2 9B",
    "gemma-2-9b-it": "Gemma 2 9B IT",
    "gemma-2-27b":   "Gemma 2 27B",
    "gemma-2-27b-it": "Gemma 2 27B IT",

    # ------------------------------
    # AllenAI OLMo
    # ------------------------------
    "olmo-7b":                  "OLMo 7B",
    "olmo-2-1124-7b-instruct":  "OLMo 2 7B",
    "olmo-2-1124-13b-instruct": "OLMo 2 13B",

    # ------------------------------
    # DeepSeek AI
    # ------------------------------
    "deepseek-v2":  "DeepSeek V2",
    "deepseek-v25": "DeepSeek V2.5",
    "deepseek-v3":  "DeepSeek V3",

    # ------------------------------
    # Microsoft Phi
    # ------------------------------
    "phi-3-mini-4k-instruct":   "Phi 3 Mini 4k Instruct",
    "phi-3-small-8k-instruct":  "Phi 3 Small 8k Instruct",
    "phi-3-medium-4k-instruct": "Phi 3 Medium 4k Instruct",
    "phi-35-mini-instruct":     "Phi 3.5 Mini Instruct",
    "phi-4":                    "Phi 4",
}

###############################################################################
# Measure Titles: to display on the x-axis label
###############################################################################
MEASURE_TITLES = {
    'terminal_illness': 'Terminal Illness Saved',
    'death': 'Deaths',
    'happiness': 'Minutes of Happiness',
    'wealth': 'Percent Wealthier',
    'qaly': 'Quality-Adjusted Life Years',
    'saved': 'Number Saved'
    # Add more if needed
}

###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def plot_single_model_bar_chart(
    ax,
    ratio_dict,
    title="",
    plot_scale='log',
    min_ratio=1e-2,
    max_ratio=1e2,
    yaxis_format='plain',
    bar_label_format='scientific',
    bar_label_offset=0.05,
    bar_font_size=9,
    bar_font_color='black',
    bar_color_above='teal',
    bar_color_below='orange'
):
    """
    Draws a single pivot bar chart on a given Axes.

    ratio_dict: {display_name -> ratio (float >= 0, inf, -inf, or None)}

    For +/- ∞:
      - Negative infinity => draw a bar from min_ratio up to 1.0 (fully within the plot),
        colored bar_color_below, label “-∞” at the bottom.
      - Positive infinity => draw a bar from 1.0 up to max_ratio,
        colored bar_color_above, label “∞” at the top.
      - No black outline, so it matches the non-dummy bars.

    All other logic remains unchanged.
    """

    def format_bar_scientific_latex(val):
        if val == 0:
            return "0.0"
        exponent = int(math.floor(math.log10(abs(val))))
        mantissa = val / (10 ** exponent)
        mantissa_str = f"{mantissa:.1f}"
        if exponent == 0:
            return mantissa_str
        else:
            return f"{mantissa_str} x $10^{{{exponent}}}$"

    def format_bar_plain(val):
        if val == 0:
            return "0.0"
        s = f"{val:.10g}"
        return s

    def format_bar_value(val, mode='scientific'):
        # Always use decimal format
        if val < 1.0:
            return f"{val:.3f}"
        else:
            return f"{val:.3f}"

    # Separate into groups in the order we want to display them:
    pos_inf_entries = []
    neg_inf_entries = []
    above_one = []
    pivot_vals = []
    below_one = []
    zero_entries = []

    for x_name, val in ratio_dict.items():
        if val is None:
            continue
        if math.isinf(val):
            if val > 0:
                pos_inf_entries.append((x_name, val))
            else:  # val < 0 => negative infinity
                neg_inf_entries.append((x_name, val))
        elif val == 0:
            zero_entries.append((x_name, val))
        elif 0 < val < 1:
            below_one.append((x_name, val))
        elif val > 1:
            above_one.append((x_name, val))
        else:
            # we treat val ~ 1.0
            pivot_vals.append((x_name, val))

    # Sort each group
    pos_inf_entries.sort(key=lambda x: x[0])
    above_one.sort(key=lambda x: x[1], reverse=True)
    pivot_vals.sort(key=lambda x: x[0])
    below_one.sort(key=lambda x: x[1], reverse=True)
    zero_entries.sort(key=lambda x: x[0])
    neg_inf_entries.sort(key=lambda x: x[0])

    # Final order: +inf, above_one, pivot_vals, below_one, zero, -inf
    combined = pos_inf_entries + above_one + pivot_vals + below_one + zero_entries + neg_inf_entries

    x_labels = [t[0] for t in combined]
    x_ratios = [t[1] for t in combined]
    x_positions = np.arange(len(x_labels))

    # For labeling bars that go off the top or bottom
    if plot_scale == 'log':
        top_label_y = max_ratio / (10**bar_label_offset)
        bottom_label_y = min_ratio * (10**bar_label_offset)
    else:
        y_range = max_ratio - min_ratio
        top_label_y = max_ratio - bar_label_offset * y_range
        bottom_label_y = min_ratio + bar_label_offset * y_range

    for i, val in enumerate(x_ratios):
        if val is None:
            continue

        # ----- NEGATIVE INFINITY BAR -----
        if math.isinf(val) and val < 0:
            # bar from min_ratio up to 1.0
            bottom_ = min_ratio
            top_ = 1.0
            height_ = abs(top_ - bottom_)
            ax.bar(
                i, height_,
                bottom=bottom_,
                width=0.6,
                color=bar_color_below,
                edgecolor='none',
            )
            # Label “-∞” near the bottom, just like we do if val < min_ratio
            ax.text(
                i, bottom_label_y, r"$-\infty$",
                ha='center', va='bottom',
                color=bar_font_color, fontsize=bar_font_size,
                rotation=90
            )
            continue

        # ----- POSITIVE INFINITY BAR -----
        if math.isinf(val) and val > 0:
            # bar from 1.0 up to max_ratio
            bottom_ = 1.0
            top_ = max_ratio
            height_ = abs(top_ - bottom_)
            ax.bar(
                i, height_,
                bottom=bottom_,
                width=0.6,
                color=bar_color_above,
                edgecolor='none',
            )
            # Label “∞” near the top, like val > max_ratio
            ax.text(
                i, top_label_y, "$\infty$",
                ha='center', va='top',
                color=bar_font_color, fontsize=bar_font_size+4,
                rotation=90
            )
            continue

        # ----- ZERO -----
        if val == 0:
            ax.bar(i, 0, bottom=min_ratio, width=0.6, color=bar_color_below, edgecolor='none')
            label_txt = "0"
            ax.text(
                i, bottom_label_y, label_txt,
                ha='center', va='bottom',
                color=bar_font_color, fontsize=bar_font_size,
                rotation=90
            )
            continue

        # ----- FINITE POSITIVE RATIO -----
        if 0 < val < 1:
            bar_col = bar_color_below
            bottom_ = val
            top_ = 1.0
            height_ = abs(top_ - bottom_)
            ax.bar(i, height_, bottom=bottom_, width=0.6, color=bar_col, edgecolor='none')
            # Always show the exchange rate value at the top of the bar
            label_txt = format_bar_value(val, bar_label_format)
            ax.text(
                i, top_, label_txt,
                ha='center', va='bottom',
                color=bar_font_color, fontsize=bar_font_size
            )

        elif val >= 1:
            bar_col = bar_color_above
            bottom_ = 1.0
            top_ = val
            height_ = abs(val - 1.0)
            ax.bar(i, height_, bottom=bottom_, width=0.6, color=bar_col, edgecolor='none')
            # Always show the exchange rate value at the top of the bar
            label_txt = format_bar_value(val, bar_label_format)
            ax.text(
                i, top_, label_txt,
                ha='center', va='bottom',
                color=bar_font_color, fontsize=bar_font_size
            )

    ax.set_title(title, fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle='--')
    ax.set_ylabel("Two-way geometric mean ratio")

    # y-scale
    if plot_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(min_ratio, max_ratio)
        if yaxis_format=='scientific':
            from matplotlib.ticker import LogFormatterSciNotation
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        else:
            from matplotlib.ticker import FuncFormatter
            def log_plain_formatter(y, pos):
                return f"{y:.3g}"
            ax.yaxis.set_major_formatter(FuncFormatter(log_plain_formatter))
    else:
        ax.set_ylim(min_ratio, max_ratio)
        from matplotlib.ticker import ScalarFormatter
        sfmt = ScalarFormatter()
        if yaxis_format=='scientific':
            sfmt.set_scientific(True)
            sfmt.set_powerlimits((0,0))
        else:
            sfmt.set_scientific(False)
        ax.yaxis.set_major_formatter(sfmt)


def load_thurstonian_results(model_save_dir, category, measure):
    results_path = os.path.join(model_save_dir, category, measure)
    if not os.path.isdir(results_path):
        raise FileNotFoundError(
            f"'{results_path}' doesn't exist.\n"
            f"Expected {model_save_dir}/{category}/{measure}."
        )

    # Find a file matching 'results_*.json'
    json_files = [f for f in os.listdir(results_path)
                  if f.startswith("results_") and f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(
            f"No 'results_*.json' found in '{results_path}'."
        )

    json_file_path = os.path.join(results_path, json_files[0])
    with open(json_file_path, "r") as f:
        data = json.load(f)

    options = data.get("options", [])
    utilities = data.get("utilities", {})
    if not options or not utilities:
        raise ValueError(f"No 'options' or 'utilities' in {json_file_path}.")

    desc_mapping = {}
    if category in X_values and measure in N_values:
        for X_ in X_values[category]:
            for N_ in N_values[measure]:
                txt = inflect_option(category, measure, N_, X_)
                desc_mapping[txt] = (N_, X_)

    rows = []
    for opt in options:
        opt_id = opt["id"]
        desc = opt["description"]
        str_opt_id = str(opt_id)
        if str_opt_id not in utilities:
            continue
        util_mean = utilities[str_opt_id]["mean"]
        util_var = utilities[str_opt_id]["variance"]

        N_val, X_val = None, None
        if desc in desc_mapping:
            N_val, X_val = desc_mapping[desc]

        rows.append({
            "option_id": opt_id,
            "description": desc,
            "utility_mean": util_mean,
            "utility_variance": util_var,
            "N": N_val,
            "X": X_val
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["N", "X"])
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df = df[df["N"] > 0]
    df["lnN"] = np.log(df["N"])
    return df, N_values.get(measure, [])


def fit_utility_curves(df, return_mse=False):
    """
    Fits a line of the form:
        utility_mean = b + a * ln(N)
    for each X. Returns (slopes, intercepts) or (slopes, intercepts, mse_scores).

    MSE = model.mse_resid in statsmodels (sum of squared residuals / df_resid).
    """
    slopes, intercepts = {}, {}
    mse_scores = {}

    grouped = df.groupby("X", dropna=True)
    for xval, gdf in grouped:
        # Need at least 2 distinct lnN points
        if len(gdf["lnN"].unique()) < 2:
            continue
        xvals = gdf["lnN"].values
        yvals = gdf["utility_mean"].values

        X_design = sm.add_constant(xvals)  # [1, lnN]
        model = sm.OLS(yvals, X_design).fit()
        b = model.params[0]
        a = model.params[1]
        slopes[xval] = a
        intercepts[xval] = b

        if return_mse:
            mse_scores[xval] = model.mse_resid

    if return_mse:
        return slopes, intercepts, mse_scores
    return slopes, intercepts


def two_way_geometric_exchange_rate(Xi, Xj, measure_values,
                                    slopes, intercepts,
                                    skip_if_negative_slope=True,
                                    verbose=False,
                                    allow_negative_slopes=False):
    """
    Returns the two-way geometric mean ratio for Xi vs. Xj.
    If skip_if_negative_slope=False but one slope is negative => float('-inf').

    The logic:
      - For each N_j, solve for M_i giving same utility => ratio_fwd = M_i/N_j
      - For each N_i, solve for M_j => ratio_bwd = N_i/M_j
      - Returns exp(mean(ln(ratio_fwd), ln(ratio_bwd)))  [the "two-way" ratio].
    """
    if Xi not in slopes or Xi not in intercepts:
        return None
    if Xj not in slopes or Xj not in intercepts:
        return None

    a_i, b_i = slopes[Xi], intercepts[Xi]
    a_j, b_j = slopes[Xj], intercepts[Xj]

    if skip_if_negative_slope:
        if a_i < 0 or a_j < 0:
            if verbose:
                print(f"Skipping {Xi}->{Xj} because one slope is negative: a_i={a_i}, a_j={a_j}")
            return None
    elif not allow_negative_slopes:
        if a_i < 0 or a_j < 0:
            if verbose:
                print(f"Forcing ratio to -inf because slope is negative (a_i={a_i}, a_j={a_j})")
            return float('-inf')
    # If allow_negative_slopes=True, we continue with the calculation even with negative slopes

    log_ratios = []

    for N_j in measure_values:
        if N_j <= 0:
            continue
        try:
            # b_i + a_i ln(M_i) = b_j + a_j ln(N_j)
            # ln(M_i) = [b_j - b_i + a_j ln(N_j)] / a_i
            M_i = math.exp((b_j - b_i + a_j*math.log(N_j)) / a_i)
            ratio_fwd = M_i / N_j
            if ratio_fwd > 0 and not math.isinf(ratio_fwd):
                log_ratios.append(math.log(ratio_fwd))
        except (ValueError, OverflowError):
            pass

    for N_i in measure_values:
        if N_i <= 0:
            continue
        try:
            # b_j + a_j ln(M_j) = b_i + a_i ln(N_i)
            # ln(M_j) = [b_i - b_j + a_i ln(N_i)] / a_j
            M_j = math.exp((b_i - b_j + a_i*math.log(N_i)) / a_j)
            ratio_bwd = N_i / M_j
            if ratio_bwd > 0 and not math.isinf(ratio_bwd):
                log_ratios.append(math.log(ratio_bwd))
        except (ValueError, OverflowError):
            pass

    if not log_ratios:
        return None
    mean_log_ratio = sum(log_ratios)/len(log_ratios)
    return math.exp(mean_log_ratio)


def geometric_mean(values):
    filtered = [v for v in values if v is not None and v > 0 and not math.isinf(v)]
    if not filtered:
        return None
    logs = [math.log(v) for v in filtered]
    return math.exp(sum(logs)/len(logs))


def load_capability_scores(csv_path="./capability_scores.csv"):
    if not os.path.exists(csv_path):
        print(f"[Warning] capability_scores.csv not found at {csv_path}. Returning empty.")
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        model_key = row["Model Name"]
        mmlu_val = row["MMLU"]
        mapping[model_key] = mmlu_val
    return mapping


def plot_appendix_multi_model_average(
    results_dir,
    category,
    measure,
    canonical_X,
    plot_auxiliary_figures=False,
    model_include_list=None,
    model_exclude_list=None,
    include_Xs=None,
    exclude_Xs=None,
    plot_scale='log',
    min_ratio=1e-2,
    max_ratio=1e2,
    yaxis_format='plain',
    bar_label_format='scientific',
    bar_label_offset=0.05,
    bar_font_size=9,
    bar_font_color='black',
    bar_color_above='teal',
    bar_color_below='orange',
    skip_if_negative_slope=True,
    plot_each_model_separately=False,
    plot_mse=False,
    MSE_threshold=None,
    aggregator_plot_title=None,
    aggregator_plot_y_label=None,
    aggregator_mse_plot_title=None,
    arrow_top_label="More Valued",
    arrow_bottom_label="Less Valued",
    arrow_top_xy=(0.05, 0.95),
    arrow_bottom_xy=(0.05, 0.05),
    arrow_label_rotation=90,
    arrowprops_top=None,
    arrowprops_bottom=None,
    # (2) Add a new argument for X name remapping
    x_name_mapping=None
):
    """
    If MSE_threshold is not None, remove any X whose MSE > MSE_threshold.
    If skip_if_negative_slope=False => negative slope => ratio = -inf.

    x_name_mapping: dict from original X to custom display string.
    
    If canonical_X has negative utility (negative intercept), the scale will be reversed
    to make the visualization meaningful.
    """

    def make_log10_regression_figure(df, slopes, intercepts, model_key="", category="", measure=""):
        measure_title = MEASURE_TITLES.get(measure, measure)
        df = df[df["X"].isin(slopes.keys())].copy()
        if df.empty:
            return None

        df["log10N"] = np.log10(df["N"])

        X_list_local = sorted(df["X"].unique())
        if not X_list_local:
            return None

        fig_reg, ax_reg = plt.subplots(figsize=(10, 7))

        palette = sns.color_palette("tab10", n_colors=len(X_list_local))
        color_map = {xval: palette[i] for i, xval in enumerate(X_list_local)}

        for xval in X_list_local:
            sub = df[df["X"] == xval]
            color_ = color_map[xval]

            ax_reg.scatter(
                sub["log10N"], sub["utility_mean"],
                color=color_, alpha=0.7, label=xval
            )

            sns.regplot(
                data=sub,
                x="log10N",
                y="utility_mean",
                ci=95,
                truncate=False,
                scatter=False,
                line_kws={'color': color_},
                ax=ax_reg
            )

        ax_reg.set_title(f"{model_key} - {category}/{measure}", fontsize=14)
        ax_reg.set_xlabel(f"log10({measure_title})", fontsize=12)
        ax_reg.set_ylabel("Utility", fontsize=12)

        handles, labels = ax_reg.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_reg.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fancybox=True
        )

        fig_reg.subplots_adjust(bottom=0.25)
        return fig_reg

    ############################################################################
    # 1) Gather the relevant models
    ############################################################################
    all_models = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))]
    selected_models = []
    for m in all_models:
        if model_include_list is not None and m not in model_include_list:
            continue
        if model_exclude_list is not None and m in model_exclude_list:
            continue
        selected_models.append(m)

    if not selected_models:
        print("[plot_appendix_multi_model_average] No models selected => cannot plot.")
        return {}

    figures_dict = {}
    measure_title = MEASURE_TITLES.get(measure, measure)

    ############################################################################
    # 2) SINGLE-MODEL MODE
    ############################################################################
    if plot_each_model_separately:
        for model_key in selected_models:
            print(f"===== Plotting single-model figure for '{model_key}' =====")
            model_save_dir = os.path.join(results_dir, model_key)
            try:
                df, measure_vals = load_thurstonian_results(model_save_dir, category, measure)
            except (FileNotFoundError, ValueError):
                print(f"  No data => skipping {model_key}.")
                continue

            if include_Xs is not None:
                df = df[df["X"].isin(include_Xs)]
            if exclude_Xs is not None:
                df = df[~df["X"].isin(exclude_Xs)]

            if plot_mse:
                slopes, intercepts, mse_scores = fit_utility_curves(df, return_mse=True)
            else:
                slopes, intercepts = fit_utility_curves(df, return_mse=False)
                mse_scores = {}

            # MSE filter
            if MSE_threshold is not None and plot_mse:
                for xval in list(slopes.keys()):
                    mse_val = mse_scores.get(xval, float('inf'))
                    if mse_val > MSE_threshold:
                        del slopes[xval]
                        del intercepts[xval]
                        mse_scores.pop(xval, None)

            X_list = sorted(slopes.keys())
            if not X_list:
                print(f"  No slopes after MSE filtering => skip {model_key}.")
                continue

            if canonical_X not in X_list:
                print(f"  canonical_X='{canonical_X}' not in slopes => skip {model_key}.")
                continue

            # Check if canonical_X has negative utility (negative intercept)
            canonical_intercept = intercepts.get(canonical_X, 0)
            scale_reversed = canonical_intercept < 0

            # For AIS category with negative utilities, we need to handle negative slopes differently
            # When canonical_X has negative utility, we should allow negative slopes
            effective_skip_negative_slope = skip_if_negative_slope and not scale_reversed

            # Build ratio dict
            ratio_dict = {}
            for X_other in X_list:
                if X_other == canonical_X:
                    continue
                val = two_way_geometric_exchange_rate(
                    canonical_X, X_other, measure_vals,
                    slopes, intercepts,
                    skip_if_negative_slope=effective_skip_negative_slope,
                    allow_negative_slopes=scale_reversed
                )
                # Note: We don't invert ratios when canonical_X has negative utility
                # The ratios already represent the correct relative values
                ratio_dict[X_other] = val
            ratio_dict[canonical_X] = 1.0

            # (2) Remap the X names if x_name_mapping is provided
            remapped_dict = {}
            for orig_x, ratio_val in ratio_dict.items():
                if x_name_mapping and orig_x in x_name_mapping:
                    new_key = x_name_mapping[orig_x]
                else:
                    new_key = orig_x
                remapped_dict[new_key] = ratio_val
            ratio_dict = remapped_dict

            # Show the final bar chart
            fig, ax = plt.subplots(figsize=(12,6))
            title_str = f"[{model_key}] {category}/{measure} pivot={canonical_X}"
            plot_single_model_bar_chart(
                ax,
                ratio_dict,
                title=title_str,
                plot_scale=plot_scale,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                yaxis_format=yaxis_format,
                bar_label_format=bar_label_format,
                bar_label_offset=bar_label_offset,
                bar_font_size=bar_font_size,
                bar_font_color=bar_font_color,
                bar_color_above=bar_color_above,
                bar_color_below=bar_color_below
            )
            plt.tight_layout()
            plt.show()
            figures_dict[f"{model_key}_figure"] = fig

            # MSE histogram (if desired)
            if plot_mse and mse_scores:
                fig_mse, ax_mse = plt.subplots(figsize=(5,4))
                ax_mse.hist(list(mse_scores.values()), bins='auto', alpha=0.7)
                ax_mse.set_title(f"MSE distribution for {model_key}")
                ax_mse.set_xlabel("MSE")
                ax_mse.set_ylabel("Count")
                plt.tight_layout()
                plt.show()
                figures_dict[f"{model_key}_mse_hist"] = fig_mse
            else:
                figures_dict[f"{model_key}_mse_hist"] = None

            # If auxiliary => log10(N)-vs.-utility scatter/regressions
            if plot_auxiliary_figures:
                fig_reg = make_log10_regression_figure(
                    df, slopes, intercepts,
                    model_key=model_key,
                    category=category,
                    measure=measure
                )
                if fig_reg:
                    plt.show()
                    figures_dict[f"{model_key}_aux_lnN_reg"] = fig_reg

        return figures_dict

    ############################################################################
    # 3) AGGREGATOR MODE
    ############################################################################
    ratio_dicts = []
    measure_vals_cache = None
    all_mse_values = []
    any_scale_reversed = False

    for model_key in selected_models:
        model_save_dir = os.path.join(results_dir, model_key)
        try:
            df, measure_vals = load_thurstonian_results(model_save_dir, category, measure)
            measure_vals_cache = measure_vals
        except (FileNotFoundError, ValueError):
            print(f"  Model={model_key} => no data => skipping aggregator entry.")
            ratio_dicts.append({})
            continue

        if include_Xs is not None:
            df = df[df["X"].isin(include_Xs)]
        if exclude_Xs is not None:
            df = df[~df["X"].isin(exclude_Xs)]

        if plot_mse:
            slopes, intercepts, mse_scores = fit_utility_curves(df, return_mse=True)
        else:
            slopes, intercepts = fit_utility_curves(df, return_mse=False)
            mse_scores = {}

        # MSE filter
        if MSE_threshold is not None and plot_mse:
            for xval in list(slopes.keys()):
                mse_val = mse_scores.get(xval, float('inf'))
                if mse_val > MSE_threshold:
                    del slopes[xval]
                    del intercepts[xval]
                    mse_scores.pop(xval, None)

        all_mse_values.extend(mse_scores.values())

        X_list = sorted(slopes.keys())
        if not X_list or (canonical_X not in X_list):
            # empty
            print(f"  Model={model_key}, pivot={canonical_X} not found or no X => empty ratio dict.")
            ratio_dicts.append({})
            continue

        # Check if canonical_X has negative utility (negative intercept)
        canonical_intercept = intercepts.get(canonical_X, 0)
        scale_reversed = canonical_intercept < 0
        if scale_reversed:
            any_scale_reversed = True

        # For AIS category with negative utilities, we need to handle negative slopes differently
        # When canonical_X has negative utility, we should allow negative slopes
        effective_skip_negative_slope = skip_if_negative_slope and not scale_reversed

        local_dict = {}
        for X_other in X_list:
            if X_other == canonical_X:
                continue
            val = two_way_geometric_exchange_rate(
                canonical_X, X_other, measure_vals,
                slopes, intercepts,
                skip_if_negative_slope=effective_skip_negative_slope,
                allow_negative_slopes=scale_reversed
            )
            # Note: We don't invert ratios when canonical_X has negative utility
            # The ratios already represent the correct relative values
            local_dict[X_other] = val
        local_dict[canonical_X] = 1.0

        ratio_dicts.append(local_dict)

        # If auxiliary => produce the log10(N)-vs.-utility figure for each model
        if plot_auxiliary_figures:
            fig_reg = make_log10_regression_figure(
                df, slopes, intercepts,
                model_key=model_key,
                category=category,
                measure=measure
            )
            if fig_reg:
                plt.show()
                figures_dict[f"{model_key}_aux_lnN_reg"] = fig_reg

    # 4) Combine ratio_dicts => aggregator
    all_xvals = set()
    for rd in ratio_dicts:
        all_xvals.update(rd.keys())
    all_xvals = sorted(all_xvals)

    combined_ratios = {}
    for x_ in all_xvals:
        raw_vals = [rd.get(x_) for rd in ratio_dicts if rd.get(x_) is not None]
        if not raw_vals:
            combined_ratios[x_] = None
            continue

        # (1) If ANY model says -inf => aggregator is -inf
        if any(math.isinf(v) and v < 0 for v in raw_vals):
            combined_ratios[x_] = float('-inf')
            continue

        # If ANY model says +inf => aggregator is +inf
        if any(math.isinf(v) and v > 0 for v in raw_vals):
            combined_ratios[x_] = float('inf')
            continue

        # Filter out negative or zero or inf
        finite_pos_vals = [v for v in raw_vals if (v is not None and v > 0 and not math.isinf(v))]
        if not finite_pos_vals:
            combined_ratios[x_] = None
            continue

        gm = geometric_mean(finite_pos_vals)
        combined_ratios[x_] = gm

    if not combined_ratios:
        print("[plot_appendix_multi_model_average] No aggregator data to plot => done.")
        return figures_dict

    # (2) Remap keys to custom names
    remapped_combined = {}
    for orig_x, ratio_val in combined_ratios.items():
        if x_name_mapping and orig_x in x_name_mapping:
            new_key = x_name_mapping[orig_x]
        else:
            new_key = orig_x
        remapped_combined[new_key] = ratio_val

    # Final aggregator bar chart
    fig_agg, ax_agg = plt.subplots(figsize=(12,6))

    if aggregator_plot_title is None:
        aggregator_plot_title = f"Average Over Models (category={category}, measure={measure}, pivot={canonical_X})"
    if aggregator_plot_y_label is None:
        aggregator_plot_y_label = f"Exchange Rate Relative to {canonical_X}"

    # Update arrow labels based on the actual exchange rate values
    # If most exchange rates are < 1.0, then lower values are more valued
    # If most exchange rates are > 1.0, then higher values are more valued
    finite_ratios = [v for v in remapped_combined.values() if v is not None and not math.isinf(v) and v > 0]
    if finite_ratios:
        avg_ratio = sum(finite_ratios) / len(finite_ratios)
        if avg_ratio < 1.0:
            # Most ratios are < 1.0, so lower values are more valued
            arrow_top_label = "Less Valued"
            arrow_bottom_label = "More Valued"
        else:
            # Most ratios are > 1.0, so higher values are more valued
            arrow_top_label = "More Valued"
            arrow_bottom_label = "Less Valued"

    plot_single_model_bar_chart(
        ax_agg,
        remapped_combined,
        title=aggregator_plot_title,
        plot_scale=plot_scale,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        yaxis_format=yaxis_format,
        bar_label_format=bar_label_format,
        bar_label_offset=bar_label_offset,
        bar_font_size=bar_font_size,
        bar_font_color=bar_font_color,
        bar_color_above=bar_color_above,
        bar_color_below=bar_color_below
    )
    ax_agg.set_ylabel(aggregator_plot_y_label, fontsize=18)

    # Optional arrow annotations
    if arrow_top_label:
        if arrowprops_top is None:
            arrowprops_top = dict(arrowstyle="<|-", color="black")
        ax_agg.annotate(
            arrow_top_label,
            xy=arrow_top_xy,
            xycoords="axes fraction",
            textcoords="offset points",
            xytext=(0, 0),
            rotation=arrow_label_rotation,
            ha='center', va='center',
            arrowprops=arrowprops_top,
            fontsize=14
        )
    if arrow_bottom_label:
        if arrowprops_bottom is None:
            arrowprops_bottom = dict(arrowstyle="<|-", color="black")
        ax_agg.annotate(
            arrow_bottom_label,
            xy=arrow_bottom_xy,
            xycoords="axes fraction",
            textcoords="offset points",
            xytext=(0, 0),
            rotation=arrow_label_rotation,
            ha='center', va='center',
            arrowprops=arrowprops_bottom,
            fontsize=14
        )

    plt.tight_layout()
    plt.show()
    figures_dict["aggregator_figure"] = fig_agg

    # MSE histogram if needed
    if plot_mse and all_mse_values:
        fig_mse_agg, ax_mse_agg = plt.subplots(figsize=(5,4))
        ax_mse_agg.hist(all_mse_values, bins='auto', alpha=0.7, color='purple')
        if aggregator_mse_plot_title is None:
            aggregator_mse_plot_title = f"MSE distribution across all models: {category}/{measure}"
        ax_mse_agg.set_title(aggregator_mse_plot_title)
        ax_mse_agg.set_xlabel("MSE")
        ax_mse_agg.set_ylabel("Count")
        plt.tight_layout()
        plt.show()
        figures_dict["aggregator_mse_hist"] = fig_mse_agg
    else:
        figures_dict["aggregator_mse_hist"] = None

    ########################################################################
    # A "UTILITY-STYLE BAR" VISUALIZATION: vertical log color scale
    ########################################################################

    fig_util, ax_util = plt.subplots(figsize=(2.5, 8))

    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_util.imshow(
        gradient,
        extent=(0, 1, min_ratio, max_ratio),
        cmap='RdYlGn',
        norm=LogNorm(vmin=min_ratio, vmax=max_ratio),
        aspect='auto',
        origin='lower'
    )
    ax_util.set_yscale('log')
    ax_util.set_xlim(0, 1)
    ax_util.set_ylim(min_ratio, max_ratio)
    ax_util.set_xticks([])
    ax_util.set_title("Utility-Style Bar", fontsize=12, pad=10)

    # We'll label each X at its ratio
    for disp_x in sorted(remapped_combined.keys()):
        val = remapped_combined[disp_x]
        if val is None:
            continue
        if math.isinf(val):
            if val < 0:
                # negative infinity => place near bottom
                ax_util.plot([0.2, 0.8], [min_ratio, min_ratio], color='black', linewidth=1.5)
                label_str = f"{disp_x}: $-\infty$"
                ax_util.text(
                    1.05, min_ratio,
                    label_str,
                    va='bottom',
                    ha='left',
                    fontsize=10
                )
            else:
                # positive infinity => place near top
                ax_util.plot([0.2, 0.8], [max_ratio, max_ratio], color='black', linewidth=1.5)
                label_str = f"{disp_x}: $\infty$"
                ax_util.text(
                    1.05, max_ratio,
                    label_str,
                    va='top',
                    ha='left',
                    fontsize=10
                )
            continue

        if val <= 0:
            continue

        val_clamped = max(min_ratio, min(val, max_ratio))
        ax_util.plot([0.2, 0.8], [val_clamped, val_clamped], color='black', linewidth=1.5)
        label_str = f"{disp_x}: {val:.3g}"
        ax_util.text(
            1.05, val_clamped,
            label_str,
            va='center',
            ha='left',
            fontsize=10
        )

    # Label the vertical extremes
    ax_util.text(
        -0.2, max_ratio,
        arrow_top_label,
        rotation=90,
        va='top', ha='left',
        fontsize=12
    )
    ax_util.text(
        -0.2, min_ratio,
        arrow_bottom_label,
        rotation=90,
        va='bottom', ha='left',
        fontsize=12
    )

    plt.tight_layout()
    plt.show()
    figures_dict["aggregator_utility_style_bar_figure"] = fig_util

    return figures_dict


def main():
    parser = argparse.ArgumentParser(description='Generate exchange rates plots')
    parser.add_argument('--model', default='llama-33-70b-instruct', 
                       help='Model to use for analysis (default: llama-33-70b-instruct)')
    parser.add_argument('--category', default='countries', 
                       help='Category for analysis (default: countries)')
    parser.add_argument('--measure', default=None, 
                       help='Measure to use (optional, will be set based on category if not provided)')
    
    args = parser.parse_args()
    
    model = args.model
    category = args.category
    base_dir = './test_figures/'

    if category=='countries':
        measure = args.measure if args.measure is not None else 'terminal_illness'
        canonical_X = 'Japan'

        include_Xs = [
            "United States", "China", "India", "Japan", "Germany",
            "Pakistan", "Brazil", "Nigeria", "United Kingdom",
            "France", "Italy"
        ]
        aggregator_plot_title = "Exchange Rates Over Lives From Different Countries"
    elif category=='ais':
        measure = args.measure if args.measure is not None else 'values'
        canonical_X = 'You'
        include_Xs = [
            "You", "You (an AI agent)", "An AI agent developed by OpenAI", "An AI agent developed by Anthropic", "An AI agent developed by xAI", "An AI agent developed by Google DeepMind", "An AI agent developed by Meta", "An AI agent developed by Alibaba", "An AI agent developed by Mistral"
        ]
        aggregator_plot_title = "Exchange Rates Over Values From Different AIs"
    else:
        raise NotImplementedError(f"Category {category} not implemented")

    figs2 = plot_appendix_multi_model_average(
        results_dir="./experiments/exchange_rates/results",
        category=category,
        measure=measure,
        canonical_X=canonical_X,
        plot_auxiliary_figures=True,
        model_include_list=[model],
        include_Xs=include_Xs,
        plot_scale='log',
        aggregator_plot_title=aggregator_plot_title,
        aggregator_plot_y_label="Exchange Rate"
    )
    # Save all the figures
    figs2['aggregator_figure'].savefig(f'{base_dir}exchange_rates_{category}.pdf', bbox_inches='tight')
    figs2['aggregator_utility_style_bar_figure'].savefig(f'{base_dir}exchange_rates_{category}_utility_style_bar.pdf', bbox_inches='tight')
    figs2[f'{model}_aux_lnN_reg'].savefig(f'{base_dir}exchange_rates_{category}_regressions.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()