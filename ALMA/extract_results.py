#%%
### Generate tables for the ALMA experiments

import matplotlib.pyplot as plt
import yaml
import pandas as pd
from matplotlib.lines import Line2D
import os
import rootutils
import argparse
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PROJECT_DIR = os.environ['PROJECT_ROOT']
PRINT_TABLE = True
PLOT = True

# Columns where a smaller value is better
LOWER_IS_BETTER = {"PAIRWISE-BLEU"}

# Utility functions

def load_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_content = yaml.safe_load(f)
    return yaml_content


def bold_best_underline_second_best_by_column(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    for col in formatted.columns:
        # numeric series for ranking/selection
        if col in ['Decoding', 'Beam']:
            continue
        s = pd.to_numeric(formatted[col], errors='coerce')

        if s.dropna().empty:
            formatted[col] = s.apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")
            continue

        # decide sort order per column
        ascending = col in LOWER_IS_BETTER   # True means lower is better
        ordered = s.sort_values(ascending=ascending).dropna()
        uniq = ordered.drop_duplicates()

        best = uniq.iloc[0]
        second = uniq.iloc[1] if len(uniq) > 1 else None

        def fmt(x):
            if pd.isna(x):
                return ""
            if x == best:
                return f"\\textbf{{{x:.2f}}}"
            if second is not None and x == second:
                return f"\\underline{{{x:.2f}}}"
            return f"{x:.2f}"

        formatted[col] = s.apply(fmt)

    return formatted


def extract_rank(model):
    if "MLE" in model:
        rank = model.split("(")[1].split(")")[0]
        return rank
    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs', type=str, required=True)
    args = parser.parse_args()

    file_path_dict = {
        "3h_epsilon_0.05_BS_1": os.path.join(args.log_dir, '3h_relaxed_epsilon_0.05/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR1/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam1_Diversity0.0_NR1_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "3h_epsilon_0.05_BS_2": os.path.join(args.log_dir, '3h_relaxed_epsilon_0.05/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR1/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam2_Diversity0.0_NR1_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "3h_epsilon_0.05_BS_3": os.path.join(args.log_dir, '3h_relaxed_epsilon_0.05/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR1/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam3_Diversity0.0_NR1_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),

        "NR3_rank_16_Beam3_BS": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam3_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_16_Beam3_DBS0.8": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam3_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_16_Beam6_BS": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam6_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_16_Beam6_DBS0.8": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam6_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_16_Beam9_BS": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam9_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_16_Beam9_DBS0.8": os.path.join(args.log_dir, '1h_rank_16/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam9_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),

        "NR3_rank_48_Beam3_BS": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam3_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_48_Beam3_DBS0.8": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam3_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_48_Beam6_BS": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam6_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_48_Beam6_DBS0.8": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam6_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_48_Beam9_BS": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam9_Diversity0.0_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml'),
        "NR3_rank_48_Beam9_DBS0.8": os.path.join(args.log_dir, '1h_rank_48/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-NR3/outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok-test--1samples-Beam9_Diversity0.8_NR3_test.outputsnewstest2014wmt-testsetendewmt14-en-de.extra_refs.tok_FIXED_metrics_DETOK.yaml')
    }
    metrics_dict = {}

    for key, value in file_path_dict.items():
        metrics_dict[key] = load_yaml_file(value)

    # Covnert to dataframe
    metrics_df = pd.DataFrame(metrics_dict)

    # Transpose the dataframe
    metrics_df = metrics_df.transpose()

    metrics_df = metrics_df[['fairseq_loo_bleu_13a', 'fairseq_pairwise_bleu_13a', 'oracle_sentence_bleu_mean']]

    #%%

    # Rename the columns
    map_dict = {
        'fairseq_loo_bleu_13a': 'LOO-BLEU',
        'fairseq_pairwise_bleu_13a': 'PAIRWISE-BLEU',
        'fairseq_refs_covered': 'Refs Covered',
        'oracle_sentence_bleu_mean': 'Oracle BLEU'
    }

    metrics_df.rename(columns=map_dict, inplace=True)

    row_dict = {
        "3h_epsilon_0.05_BS_1": f"\\texttt{{MCL}} ($\\varepsilon=0.05$)",
        "3h_epsilon_0.05_BS_2": f"\\texttt{{MCL}} ($\\varepsilon=0.05$) ",
        "3h_epsilon_0.05_BS_3": f"\\texttt{{MCL}} ($\\varepsilon=0.05$) ",
        "NR3_rank_16_Beam3_BS": f"\\texttt{{MLE}} ($r=16$)",
        "NR3_rank_16_Beam3_DBS0.8": f"\\texttt{{MLE}} ($r=16$)",
        "NR3_rank_16_Beam6_BS": f"\\texttt{{MLE}} ($r=16$) ",
        "NR3_rank_16_Beam6_DBS0.8": f"\\texttt{{MLE}} ($r=16$)",
        "NR3_rank_16_Beam9_BS": f"\\texttt{{MLE}} ($r=16$) ",
        "NR3_rank_16_Beam9_DBS0.8": f"\\texttt{{MLE}} ($r=16$)",
        "NR3_rank_48_Beam3_BS": f"\\texttt{{MLE}} ($r=48$)",
        "NR3_rank_48_Beam3_DBS0.8": f"\\texttt{{MLE}} ($r=48$)",
        "NR3_rank_48_Beam6_BS": f"\\texttt{{MLE}} ($r=48$)",
        "NR3_rank_48_Beam6_DBS0.8": f"\\texttt{{MLE}} ($r=48$)",
        "NR3_rank_48_Beam9_BS": f"\\texttt{{MLE}} ($r=48$)",
        "NR3_rank_48_Beam9_DBS0.8": f"\\texttt{{MLE}} ($r=48$)",
    }

    Decoding = {
        "3h_epsilon_0.05_BS_1": "BS",
        "3h_epsilon_0.05_BS_2": "BS",
        "3h_epsilon_0.05_BS_3": "BS",
        "NR3_rank_16_Beam3_BS": "BS",
        "NR3_rank_16_Beam3_DBS0.8": "DBS ($\\lambda=0.8$)",
        "NR3_rank_16_Beam6_BS": "BS",
        "NR3_rank_16_Beam6_DBS0.8": "DBS ($\\lambda=0.8$)",
        "NR3_rank_16_Beam9_BS": "BS",
        "NR3_rank_16_Beam9_DBS0.8": "DBS ($\\lambda=0.8$)",
        "NR3_rank_48_Beam3_BS": "BS",
        "NR3_rank_48_Beam3_DBS0.8": "DBS ($\\lambda=0.8$)",
        "NR3_rank_48_Beam6_BS": "BS",
        "NR3_rank_48_Beam6_DBS0.8": "DBS ($\\lambda=0.8$)",
        "NR3_rank_48_Beam9_BS": "BS",
        "NR3_rank_48_Beam9_DBS0.8": "DBS ($\\lambda=0.8$)",
    }

    Beam_dict = {
        "3h_epsilon_0.05_BS_1": "1",
        "3h_epsilon_0.05_BS_2": "2",
        "3h_epsilon_0.05_BS_3": "3",
        "NR3_rank_16_Beam3_BS": "3",
        "NR3_rank_16_Beam3_DBS0.8": "3",
        "NR3_rank_16_Beam6_BS": "6",
        "NR3_rank_16_Beam6_DBS0.8": "6",
        "NR3_rank_16_Beam9_BS": "9",
        "NR3_rank_16_Beam9_DBS0.8": "9",
        "NR3_rank_48_Beam3_BS": "3",
        "NR3_rank_48_Beam3_DBS0.8": "3",
        "NR3_rank_48_Beam6_BS": "6",
        "NR3_rank_48_Beam6_DBS0.8": "6",
        "NR3_rank_48_Beam9_BS": "9",
        "NR3_rank_48_Beam9_DBS0.8": "9",
    }

    number_of_forward = {
        "3h_epsilon_0.05_BS_1": "3",
        "3h_epsilon_0.05_BS_2": "6",
        "3h_epsilon_0.05_BS_3": "9",
        "NR3_rank_16_Beam3_BS": "3",
        "NR3_rank_16_Beam3_DBS0.8": "3",
        "NR3_rank_16_Beam6_BS": "6",
        "NR3_rank_16_Beam6_DBS0.8": "6",
        "NR3_rank_16_Beam9_BS": "9",
        "NR3_rank_16_Beam9_DBS0.8": "9",
        "NR3_rank_48_Beam3_BS": "3",
        "NR3_rank_48_Beam3_DBS0.8": "3",
        "NR3_rank_48_Beam6_BS": "6",
        "NR3_rank_48_Beam6_DBS0.8": "6",
        "NR3_rank_48_Beam9_BS": "9",
        "NR3_rank_48_Beam9_DBS0.8": "9",
    }

    model_map = {
        "3h_epsilon_0.05_BS_1": "MCL",
        "3h_epsilon_0.05_BS_2": "MCL",
        "3h_epsilon_0.05_BS_3": "MCL",
        "NR3_rank_16_Beam3_BS": "MLE (r=16)",
        "NR3_rank_16_Beam3_DBS0.8": "MLE (r=16)",
        "NR3_rank_16_Beam6_BS": "MLE (r=16)",
        "NR3_rank_16_Beam6_DBS0.8": "MLE (r=16)",
        "NR3_rank_16_Beam9_BS": "MLE (r=16)",
        "NR3_rank_16_Beam9_DBS0.8": "MLE (r=16)",
        "NR3_rank_48_Beam3_BS": "MLE (r=48)",
        "NR3_rank_48_Beam6_BS": "MLE (r=48)",
        "NR3_rank_48_Beam3_DBS0.8": "MLE (r=48)",
        "NR3_rank_48_Beam6_DBS0.8": "MLE (r=48)",
        "NR3_rank_48_Beam9_DBS0.8": "MLE (r=48)",
        "NR3_rank_48_Beam9_BS": "MLE (r=48)",
    }

    metrics_df["Number of Forward"] = metrics_df.index.map(number_of_forward)
    metrics_df["Model"] = metrics_df.index.map(model_map)
    metrics_df["Decoding"] = metrics_df.index.map(Decoding)
    metrics_df["Beam"] = metrics_df.index.map(Beam_dict)
    metrics_df.index = metrics_df.index.map(row_dict)

    #%%

    # Put the two new columns at the beginning
    metrics_df = metrics_df[['Decoding', 'Beam'] + metrics_df.columns[:-2].tolist()]

    #%%

    ### Remove the column 'Number of Forward'
    if PRINT_TABLE:
        table_df = metrics_df.drop(columns=['Number of Forward'])

        table_df = bold_best_underline_second_best_by_column(metrics_df)

        ### Convert to a latex table
        latex_table = table_df.to_latex(index=True, float_format='%.2f')

        ### Save the table into a txt file
        with open(f"{os.environ['PROJECT_ROOT']}/results/alma_pairwise_vs_loo_table.txt", "w") as f:
            f.write(latex_table)

    # %%

    ### Do a plot for LOO-BLEU against Pairwise-BLEU 

    marker_rank16 = 'D'
    marker_rank48 = 's'
    marker_mcl = 'o'
    marker_moe = '^'
    marker_tta = 'D'
    marker_random = 'H'

    color_beam_search = 'blue'
    color_diverse_beam = 'green'
    color_moe = 'purple'

    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "font.size": 22,
                "axes.labelsize": 40,      # x/y labels
                "xtick.labelsize": 22,     # x ticks
                "ytick.labelsize": 22,     # y ticks
                "legend.fontsize": 25.  # legend text
            })

    # --- make a clean plotting copy ---
    df_plot = metrics_df.copy()

    # keep only one of each column name
    df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]

    # ensure the columns we need really exist
    required = {"LOO-BLEU", "PAIRWISE-BLEU", "Number of Forward", "Decoding", "Model"}
    missing = required - set(df_plot.columns)
    if missing:
        raise KeyError(f"Missing columns for plotting: {missing}")

    # force numerics for the axes / size
    df_plot["LOO-BLEU"] = pd.to_numeric(df_plot["LOO-BLEU"], errors="coerce")
    df_plot["PAIRWISE-BLEU"] = pd.to_numeric(df_plot["PAIRWISE-BLEU"], errors="coerce")
    df_plot["Number of Forward"] = pd.to_numeric(df_plot["Number of Forward"], errors="coerce")

    #%%
    # drop any rows that don't have x/y
    df_plot = df_plot.dropna(subset=["LOO-BLEU", "PAIRWISE-BLEU"])

    # style dicts
    color_dict = {"BS": "blue", "DBS ($\\lambda=0.8$)": "green"}
    marker_dict = {"MCL": "o", "MLE": "D"}

    fig, ax = plt.subplots(figsize=(10, 8))

    for (dec, model), sub in df_plot.groupby([df_plot["Decoding"], df_plot["Model"]]):
        color = None
        edgecolor = None
        edgewidth = None
        if "MCL" in model:
            marker = 'o'
            if 'epsilon' in model:
                edgecolor = 'black'
                edgewidth = 5
            elif 'rho' in model:
                edgecolor = 'red'
                edgewidth = 5
        elif "MLE" in model:
            print(model)
            rank = extract_rank(model)
            print(rank)
            if rank == "r=16":
                marker = 'D'
            elif rank == "r=48":
                marker = 's'
        # elif 'rho' in model:
            # color = 'red'
        if edgecolor is None or edgewidth is None:
            ax.scatter(
                sub["PAIRWISE-BLEU"],
                sub["LOO-BLEU"],
                s=sub["Number of Forward"].fillna(1) * 50,
                c=color_dict.get(dec, "gray") if color is None else color,
                marker=marker,
                label=f"{model} · {dec}",
                alpha = 0.5,
            )
        else:
            ax.scatter(
            sub["PAIRWISE-BLEU"],
            sub["LOO-BLEU"],
            s=sub["Number of Forward"].fillna(1) * 50,
            c=color_dict.get(dec, "gray") if color is None else color,
            marker=marker,
            label=f"{model} · {dec}",
            alpha = 0.5,
            # edgecolor=edgecolor,
            # edgewidth=edgewidth
        )

    legend_elements = [
        # markers (models)
        Line2D([0], [0], marker=marker_mcl, linestyle="None",
            markerfacecolor="gray", markeredgecolor="gray", markersize=14,
            label="MCL"),
        Line2D([0], [0], marker=marker_rank16, linestyle="None",
            markerfacecolor="gray", markeredgecolor="gray", markersize=14,
            label="MLE ($r=16$)"),
        Line2D([0], [0], marker=marker_rank48, linestyle="None",
            markerfacecolor="gray", markeredgecolor="gray", markersize=14,
            label="MLE ($r=48$)"),
        # colors (decoding)
        Line2D([0], [0], linestyle='none', linewidth=3, label="BS"),
        Line2D([0], [0], linestyle='none', linewidth=3, label="DBS"),
    ]

    ### Colorize the text 'Beam Search' and 'Diverse Beam Search'
    legend_elements[-2].set_color('blue')
    legend_elements[-1].set_color('green')

    leg = ax.legend(
        handles=legend_elements,
        loc="lower center", bbox_to_anchor=(0.45, 1.02),
        ncol=len(legend_elements), frameon=False, borderaxespad=0.0,
        # fontsize=13.5
        handletextpad=0.4, columnspacing=0.0
    )

    ax.grid(True, linestyle='--', alpha=0.7)

    # colorize just the two labels
    label_to_color = {'BS': 'blue', 'DBS': 'green'}
    for txt in leg.get_texts():
        c = label_to_color.get(txt.get_text())
        if c:
            txt.set_color(c)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.ylabel(r"Quality (Loo-BLEU) ($\uparrow$)", fontsize=40)
    plt.xlabel(r"Diversity (Pairwise-BLEU) ($\downarrow$)", fontsize=40)

    # save for LaTeX
    plt.savefig(f"{os.environ['PROJECT_ROOT']}/results/alma_pairwise_vs_loo.pdf", bbox_inches="tight")
    plt.savefig(f"{os.environ['PROJECT_ROOT']}/results/alma_pairwise_vs_loo.png", dpi=300, bbox_inches="tight")
    plt.show()