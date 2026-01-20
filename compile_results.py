#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any



def compile_metrics(
    args,
    concept_based_methods,
    concept_based_methods_display,
    metrics,
) -> Dict[str, Dict[str, float]]:
    """
    Compile metrics into a single structure:
    {
        "metric_name_1": { "modelA": val, "modelB": val, ... },
        "metric_name_2": { "modelA": val, "modelB": val, ... },
        ...
    }
    """

    
    results_compiled: Dict[str, Dict[str, float]] = { metric_name : {} for metric_name in metrics}
    
    sentence_sim_methods = {concept_based_methods_display[concept_based_method] : {} for concept_based_method in concept_based_methods}

    interpretability_dirs = {}

    llm_classifier_model = args.model
    dataset = args.dataset

    path_dir = Path("results/concepts_metrics")
    print(f"path_dir : {path_dir}")
    assert path_dir.exists(), f"Concepts results directory does not exist: {path_dir}"
    pattern = f"{llm_classifier_model}_ft_{dataset}"
    print(f"pattern : {pattern}")
    # Collect all file names starting with the given pattern
    candidates = [f.name for f in path_dir.iterdir() if f.name.startswith(pattern)]

    # Pre-trained model + dataset
    if not candidates:
        # New pattern
        pattern = f"{llm_classifier_model}"
        candidates = [f.name for f in path_dir.iterdir() if f.name.startswith(pattern)]
        assert candidates, f"Not found the specified combination model/dataset in the concepts results, make sure to have ran the evaluation"
        results_path_dir = Path(path_dir,candidates[0])
        dataset_pattern = f"{dataset}"
        print(results_path_dir)
        dataset_candidates = [f.name for f in results_path_dir.iterdir() if f.name.startswith(dataset_pattern)]
        assert dataset_candidates, f"Not found the specified combination model/dataset in the concepts results, make sure to have ran the evaluation"
        results_path_dir = Path(results_path_dir,dataset_candidates[0])

    # Fine-tuned model on dataset  
    else:
        select_tuned_classifier = max(candidates, key=lambda s: int(s.split('_')[-1]))
        results_path_dir = Path(path_dir,select_tuned_classifier,dataset)

    
    for concept_based_method in concept_based_methods:

        ######### Find the appropriate directory where the results of this method are saved for this llm classifier
        method_concepts_results_dir = Path(results_path_dir,concept_based_method)

        hook_layer = f"layer_{args.hook_layer}"
        hook_layer_sae = f"blocks.{args.hook_layer}"
        layer_candidates = [f.name for f in method_concepts_results_dir.iterdir() if hook_layer in f.name or hook_layer_sae in f.name]
        assert layer_candidates, f"Not found the results for the method {concept_based_method} with the specified configuration. Ensure you ran the experiments with the provided configuration"
        method_concepts_results_dir = Path(method_concepts_results_dir,layer_candidates[0])

        n_concepts = f"{args.n_concepts}_concepts"
        n_concepts_candidates = [f.name for f in method_concepts_results_dir.iterdir() if f.name.startswith(n_concepts)]
        assert n_concepts_candidates, f"Not found the results for the method {concept_based_method} with the specified configuration. Ensure you ran the experiments with the provided configuration"
        method_concepts_results_dir = Path(method_concepts_results_dir,n_concepts_candidates[0])

        method_concepts_results_dir = Path(method_concepts_results_dir,"test")

        assert method_concepts_results_dir.exists(), f"{method_concepts_results_dir} does not exist"

        ######### Completeness
        accuracy_metrics_file = Path(method_concepts_results_dir,"metrics.json")
        with open(accuracy_metrics_file, "r") as f:
            accuracy_metrics = json.load(f)
        recovery_accuracy = accuracy_metrics["Recovery accuracy"]
        results_compiled["RAcc"][concept_based_methods_display[concept_based_method]] = recovery_accuracy
        avg_act_rate = accuracy_metrics["Averaged activation frequency"]
        results_compiled["Avg. Act. Rate"][concept_based_methods_display[concept_based_method]] = avg_act_rate

        ######### Causality
        causality_metrics_file = Path(method_concepts_results_dir,"causality","metrics_ablation_averaged_results.json")
        with open(causality_metrics_file, "r") as f:
            causality_metrics = json.load(f)
        
        delta_f_cond = causality_metrics["Averaged (over features) conditional label-flip rate"]
        tvd_cond = causality_metrics["Averaged (over features) Conditional TVD"]
        results_compiled["Delta_f_cond"][concept_based_methods_display[concept_based_method]] = delta_f_cond
        results_compiled["TVD_cond"][concept_based_methods_display[concept_based_method]] = tvd_cond

        ######### Interpretability
        interpretability_dir = Path(method_concepts_results_dir,"interpretability")
        interpretability_metrics_file = Path(interpretability_dir,"interpretability_metrics.json")
        with open(interpretability_metrics_file, "r") as f:
            interpretability_metrics = json.load(f)

        ## ConceptSim
        conceptsim = interpretability_metrics["Weighted Averaged ConceptSim"]
        results_compiled["ConceptSim"][concept_based_methods_display[concept_based_method]] = conceptsim

        ## SentenceSim
        sentence_sim = interpretability_metrics["SentenceSim(k) for k from 1 to 4"]
        sentence_sim_methods[concept_based_methods_display[concept_based_method]] = sentence_sim

        interpretability_dirs[concept_based_methods_display[concept_based_method]] = interpretability_dir


    return results_compiled, sentence_sim_methods, interpretability_dirs




def to_latex_table(
    compiled: Dict[str, Dict[str, float]],
    models: List[str],
    models_display : List[str],
    caption: str,
    label: str,
    float_fmt: str = "%.3f",
) -> str:
    """
    Create a LaTeX table string with:
      - rows = metrics
      - columns = models
    """
    # Sort metrics for consistent order
    # metric_names = sorted(compiled.keys())
    metric_names = compiled.keys()


    n_models = len(models)
    # l for metric column, then one 'c' per model column
    col_spec = "l" + "c" * n_models

    header_models = " & ".join(models_display)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")
    lines.append(f"        Metric & {header_models} \\\\")
    lines.append("        \\midrule")

    for metric in metric_names:
        row_values = []
        for m in models:
            val = compiled[metric].get(m, None)
            if isinstance(val, (int, float)):
                row_values.append(float_fmt % val)
            elif val is None:
                row_values.append("--")
            else:
                row_values.append(str(val))

        row = " & ".join(row_values)
        lines.append(f"        {metric} & {row} \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


def save_latex_table(
    compiled: Dict[str, Dict[str, float]],
    models: List[str],
    models_display : List[str],
    save_dir: Path,
    args
):

    dataset = args.dataset
    llm_model = args.model

    caption = f"Evaluation metrics for {llm_model.replace('_', '-')} on {dataset.replace('_', ' ')} ."
    label = f"tab:{dataset}_metrics"

    tex = to_latex_table(
        compiled=compiled,
        models=models,
        models_display=models_display,
        caption=caption,
        label=label,
        float_fmt="%.3f",
    )

    out_tex = Path(save_dir,'metrics_table.tex')

    with out_tex.open("w") as f:
        f.write(tex)


from pathlib import Path
from typing import Dict, List

def to_markdown_table(
    compiled: Dict[str, Dict[str, float]],
    models_display: Dict[str,str],
    metrics_display: Dict[str,str],
    caption: str,
    label: str,
    float_fmt: str = "%.3f",
) -> str:
    """
    Create a Markdown table string with:
      - rows = metrics
      - columns = models

    compiled: dict[metric][model] -> value
    """

    # Keep the order of metrics as given (or wrap in list() if using Python <3.7)
    metric_names = compiled.keys()

    # Header row
    list_models_display = list(models_display.values())
    header_models = " | ".join(list_models_display)
    header = f"| Metric | {header_models} |"

    # Alignment row: left for Metric, right for numeric columns
    align_models = " | ".join(["---:" for _ in list(models_display.values())])
    align = f"|:-------| {align_models} |"

    lines = []

    # Caption + label (GitHub will just render this as text/comment)
    if caption:
        lines.append(f"**Table.** {caption}")
        lines.append("")  # blank line before the table
    if label:
        lines.append(f"<!-- {label} -->")
        lines.append("")

    lines.append(header)
    lines.append(align)

    print(compiled)

    # Data rows
    for metric in metric_names:
        row_values = []
        for m in list(models_display.values()):
            val = compiled[metric].get(m, None)
            if isinstance(val, (int, float)):
                row_values.append(float_fmt % val)
            elif val is None:
                row_values.append("--")
            else:
                row_values.append(str(val))

        row = " | ".join(row_values)
        lines.append(f"| {metrics_display[metric]} | {row} |")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def save_markdown_table(
    compiled: Dict[str, Dict[str, float]],
    models_display: Dict[str,str],
    metrics_display: Dict[str,str],
    save_dir: Path,
    args,
):
    dataset = args.dataset
    llm_model = args.model

    caption = (
        f"Evaluation metrics for `{llm_model.replace('_', '-')}` "
        f"on `{dataset.replace('_', ' ')}`."
    )
    label = f"tab:{dataset}_metrics"

    md = to_markdown_table(
        compiled=compiled,
        models_display=models_display,
        metrics_display=metrics_display,
        caption=caption,
        label=label,
        float_fmt="%.3f",
    )

    out_md = Path(save_dir, "metrics_table.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(md)



import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, AutoMinorLocator

plt.style.use('default')

def create_sentencesim_graph(
    series_dict,
    x=None,
    save_dir=None,
    filename="SentenceSim_plot.pdf"
):
   
    # Assume all lists have same length
    first_key = next(iter(series_dict))
    y_example = series_dict[first_key]

    if x is None:
        x = list(range(1, len(y_example) + 1))

    fig, ax = plt.subplots(figsize=(6.6, 4.5), facecolor='white')
    ax.set_facecolor('white')

    # Some marker shapes to cycle through
    markers = ['o', 's', '^', 'd', 'p', 'v', 'X', '*']

    for i, (name, y) in enumerate(series_dict.items()):
        marker = markers[i % len(markers)]

        # Line + marker
        ax.plot(x, y, linewidth=2, marker=marker, markevery=20)

        # Explicit scatter for clearer legend symbols
        ax.scatter(x, y, s=60, marker=marker, label=name)

    # X ticks: integers only, no minor ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x', which='minor', length=0)
    ax.grid(which='minor', axis='x', linewidth=0)

    # Grid
    ax.grid(which='major', linestyle='--', linewidth=0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    # Labels
    ax.set_xlabel(r"$k$ common top-$p$ concepts", fontsize=12)
    ax.set_ylabel(r"SentenceSim($k$)", fontsize=12)

    # Legend
    ax.legend(ncol=len(series_dict), loc="upper center", bbox_to_anchor=(0.5, 1.15))

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig, ax


def parse_args():
    p = argparse.ArgumentParser(
        description="Compile per-model metrics JSONs into one JSON + LaTeX table."
    )
    p.add_argument(
        "--model",
        type=str,
        default="pythia_410m",
        help="LLM model classifier on which the concepts based explainability methods were trained and evaluated",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="agnews",
        help="Dataset name",
    )
    p.add_argument(
        "--hook-layer",
        type=int,
        default=23,
        help="Layer depth at which the experiment was ran",
    )
    p.add_argument(
        "--n_concepts",
        type=int,
        default=20,
        help="Upper bound on the number of concepts to extract",
    )
    p.add_argument(
        "--d-sae",
        type=int,
        default=2048,
        help="Full dimenstionality of the SAE hidden size dim(z_class+z_ctx)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    concept_based_methods = ["ica","concept_shap","hi_concept","sae_logistic_regression","sae_truncation"]
    concept_based_methods_display = {"ica" : "ICA","concept_shap" : "ConceptShap","hi_concept" : "HIConcept", "sae_logistic_regression" : "SAE", "sae_truncation" : "ClassifSAE"}

    metrics = ["RAcc","Avg. Act. Rate", "Delta_f_cond", "TVD_cond", "ConceptSim"]
    metrics_display = {"RAcc" : "RAcc", "Avg. Act. Rate" : "Avg. Act. Rate", "Delta_f_cond" : "$\Delta f^{cond}$","TVD_cond" :"$TVD_{cond}$","ConceptSim" : "ConceptSim"}
    
    # Slight misalignment TO FIX
    if args.dataset is not None:
        args.dataset = "ag_news" if args.dataset == "agnews" else args.dataset
    if args.model is not None:
        m = args.model.lower()
        if "pythia" in m:
            args.model = args.model.replace("_", "-")
        if "gpt" in m:
            args.model = "gpt-j-6B"


    compiled_results,sentence_sim_methods, interpretability_dirs = compile_metrics(
       args=args,
       concept_based_methods=concept_based_methods,
       concept_based_methods_display=concept_based_methods_display,
       metrics=metrics,
    )

    root_dir = Path.cwd()
    example_directory = Path(root_dir,f"concepts_comparison_{args.model}_{args.dataset}")
    shutil.rmtree(example_directory, ignore_errors=True); example_directory.mkdir(parents=True, exist_ok=True)
    compiled_results_file = Path(example_directory,"concepts_metrics.json")

    with compiled_results_file.open("w", encoding="utf-8") as f:
        json.dump(compiled_results, f, ensure_ascii=False, indent=4)

    # Create and save SentenceSim metrics comparison graph
    create_sentencesim_graph(
        sentence_sim_methods,
        save_dir=example_directory,       
        filename="SentenceSim_comparison.pdf"  
    )

    # Save Metrics Table in Markdown
    save_markdown_table(
        compiled=compiled_results,
        models_display=concept_based_methods_display,
        metrics_display=metrics_display,
        save_dir=example_directory,
        args=args,
    )

    # Save qualitative interpretation of the discovered concepts
    qualitative_concepts_dir = Path(example_directory,"qualitative_concepts")

    for concept_based_method, interpretability_dir in interpretability_dirs.items():

        concepts_definition_dir = Path(interpretability_dir,"concepts_definition")
        if os.path.isdir(concepts_definition_dir):
            shutil.copytree(concepts_definition_dir, Path(qualitative_concepts_dir,concept_based_method,"concepts_definition"))

        concepts_wordcloud_dir = Path(interpretability_dir,"top_ngrams_tf_idf")
        if os.path.isdir(concepts_wordcloud_dir):
            shutil.copytree(concepts_wordcloud_dir, Path(qualitative_concepts_dir,concept_based_method,"concepts_wordclouds"))

    print(f"[INFO] Wrote compiled metrics JSON to {example_directory}")
    print(f"[INFO] Wrote Metrics Comparison table to {example_directory}")


if __name__ == "__main__":
    main()
