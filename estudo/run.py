"""
Orquestrador principal do estudo.
Executa em sequência:
1. Geração (ou carga) do conjunto de dados sintético
2. Treinamento e avaliação dos modelos (DT, RF, XGBoost) + baselines
3. Teste estatístico pareado e gráficos comparativos
4. Salva relatório de métricas em output/result-[nome-arquivo].md
"""

import os
import sys
import subprocess
import warnings
import pandas as pd

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:sklearn.utils.parallel,ignore::FutureWarning"

# Check if generate dataset is requested and clean sys.argv
GENERATE_DATASET = False
for arg in list(sys.argv):
    if arg in ["--generate-dataset", "--generate-datataset"]:
        GENERATE_DATASET = True
        if arg in sys.argv:
            sys.argv.remove(arg)

from dotenv import load_dotenv, dotenv_values

# Load default .env first
load_dotenv()

# If --env is specified, override with custom values (skip empty ones)
_env_file_used = ".env"
if "--env" in sys.argv:
    idx = sys.argv.index("--env")
    if idx + 1 < len(sys.argv):
        _env_file_used = sys.argv[idx + 1]
        custom_values = dotenv_values(_env_file_used)
        for key, value in custom_values.items():
            if value:
                os.environ[key] = value

from generate_data import generate_and_save, print_distribution_check, N_SAMPLES
from models import (
    run_decision_tree,
    run_random_forest,
    run_gradient_boosting,
    run_baselines,
    wilcoxon_tests,
    save_comparison_plots,
)


def get_env_variables_display() -> str:
    """Retorna todas as variáveis do .env formatadas (sem chaves/segredos)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = []

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "KEY" in line.split("=")[0].upper():
                    continue
                lines.append(line)

    return "\n".join(lines)


def execute_notebooks(timestamp: str) -> None:
    """Executa os notebooks e salva cópias com o timestamp do CSV."""
    notebooks = [
        ("models/decision_tree.ipynb", f"models/decision_tree_{timestamp}.ipynb"),
        ("models/random_forest.ipynb", f"models/random_forest_{timestamp}.ipynb"),
        ("models/gradient_boosting.ipynb", f"models/gradient_boosting_{timestamp}.ipynb"),
    ]

    for source, output in notebooks:
        name = os.path.basename(source).replace(".ipynb", "")
        print(f"  Executando {name}...")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.kernel_name=tcc-estudo",
                source,
                "--output", os.path.basename(output),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Filter out sklearn parallel warnings from error output
            stderr_lines = [
                line for line in result.stderr.splitlines()
                if "sklearn.utils.parallel" not in line and "warnings.warn" not in line
            ]
            filtered_stderr = "\n".join(stderr_lines).strip()
            if filtered_stderr:
                print(f"  AVISO: Falha ao executar {name}: {filtered_stderr}", file=sys.stderr)
            else:
                print(f"  AVISO: Falha ao executar {name} (verifique o notebook)", file=sys.stderr)
        else:
            print(f"  Salvo: {output}")


def format_baselines_section(baseline_results: dict) -> str:
    """Tabela de baselines (Dummy majoritário e Regressão Logística)."""
    lines = [
        "## Baselines de Referência",
        "",
        "| Métrica | " + " | ".join(baseline_results.keys()) + " |",
        "|---" * (len(baseline_results) + 1) + "|",
    ]
    metric_rows = [
        ("Acurácia (holdout)", lambda r: f"{r['holdout']['accuracy']*100:.2f}%"),
        ("F1-Score (holdout)", lambda r: f"{r['holdout']['f1_weighted']*100:.2f}%"),
        ("Acurácia (CV k=10)", lambda r: f"{r['cv']['accuracy']['test_mean']*100:.2f}% ± {r['cv']['accuracy']['test_std']*100:.2f}%"),
        ("F1-Score (CV k=10)", lambda r: f"{r['cv']['f1_weighted']['test_mean']*100:.2f}% ± {r['cv']['f1_weighted']['test_std']*100:.2f}%"),
    ]
    for label, fmt in metric_rows:
        lines.append(f"| **{label}** | " + " | ".join(fmt(r) for r in baseline_results.values()) + " |")
    return "\n".join(lines) + "\n\n---\n\n"


def build_output_md(dt_results: dict, rf_results: dict, gb_results: dict, data_file: str, env_vars: str, baseline_results: dict | None = None) -> str:
    """Monta o relatório .md final com cabeçalho, variáveis e tabelas de métricas."""
    dt_cv = dt_results["cv"]
    rf_cv = rf_results["cv"]
    gb_cv = gb_results["cv"]

    header = f"""# Resultado do Estudo Comparativo

**Arquivo de dados:** `{data_file}`
**Data da execução:** {os.path.basename(data_file).replace("dados_academicos_", "").replace(".csv", "")}

---

## Variáveis de Configuração (.env)

```
{env_vars}
```

---

## Tabela de Métricas

| Métrica | Árvore de Decisão | Floresta Aleatória | {gb_results['model_name']} |
|---|---|---|---|
| **Acurácia (holdout)** | {dt_results['holdout']['accuracy']*100:.2f}% | {rf_results['holdout']['accuracy']*100:.2f}% | {gb_results['holdout']['accuracy']*100:.2f}% |
| **F1-Score (holdout)** | {dt_results['holdout']['f1_weighted']*100:.2f}% | {rf_results['holdout']['f1_weighted']*100:.2f}% | {gb_results['holdout']['f1_weighted']*100:.2f}% |
| **Precision (holdout)** | {dt_results['holdout']['precision_weighted']*100:.2f}% | {rf_results['holdout']['precision_weighted']*100:.2f}% | {gb_results['holdout']['precision_weighted']*100:.2f}% |
| **Recall (holdout)** | {dt_results['holdout']['recall_weighted']*100:.2f}% | {rf_results['holdout']['recall_weighted']*100:.2f}% | {gb_results['holdout']['recall_weighted']*100:.2f}% |
| **AUC-ROC (holdout)** | {dt_results['holdout']['auc_weighted']*100:.2f}% | {rf_results['holdout']['auc_weighted']*100:.2f}% | {gb_results['holdout']['auc_weighted']*100:.2f}% |
| **Acurácia (CV k=10)** | {dt_cv['accuracy']['test_mean']*100:.2f}% ± {dt_cv['accuracy']['test_std']*100:.2f}% | {rf_cv['accuracy']['test_mean']*100:.2f}% ± {rf_cv['accuracy']['test_std']*100:.2f}% | {gb_cv['accuracy']['test_mean']*100:.2f}% ± {gb_cv['accuracy']['test_std']*100:.2f}% |
| **F1-Score (CV k=10)** | {dt_cv['f1_weighted']['test_mean']*100:.2f}% ± {dt_cv['f1_weighted']['test_std']*100:.2f}% | {rf_cv['f1_weighted']['test_mean']*100:.2f}% ± {rf_cv['f1_weighted']['test_std']*100:.2f}% | {gb_cv['f1_weighted']['test_mean']*100:.2f}% ± {gb_cv['f1_weighted']['test_std']*100:.2f}% |
| **Precision (CV k=10)** | {dt_cv['precision_weighted']['test_mean']*100:.2f}% ± {dt_cv['precision_weighted']['test_std']*100:.2f}% | {rf_cv['precision_weighted']['test_mean']*100:.2f}% ± {rf_cv['precision_weighted']['test_std']*100:.2f}% | {gb_cv['precision_weighted']['test_mean']*100:.2f}% ± {gb_cv['precision_weighted']['test_std']*100:.2f}% |
| **Recall (CV k=10)** | {dt_cv['recall_weighted']['test_mean']*100:.2f}% ± {dt_cv['recall_weighted']['test_std']*100:.2f}% | {rf_cv['recall_weighted']['test_mean']*100:.2f}% ± {rf_cv['recall_weighted']['test_std']*100:.2f}% | {gb_cv['recall_weighted']['test_mean']*100:.2f}% ± {gb_cv['recall_weighted']['test_std']*100:.2f}% |

---

"""

    baselines_section = format_baselines_section(baseline_results) if baseline_results else ""
    return header + baselines_section


def main():
    print("=" * 60)
    print("ORQUESTRADOR - ESTUDO DE MODELOS PREDITIVOS")
    print(f"Configuração: {_env_file_used}")
    print("=" * 60)

    # 1. Carregar ou Gerar conjunto de dados
    fixed_path = os.path.join("data", "dados_academicos.csv")
    if GENERATE_DATASET or not os.path.exists(fixed_path):
        if not GENERATE_DATASET:
            print("\n[!] Alerta: Dataset fixo não encontrado. Gerando um novo dataset...")
        else:
            print("\n[1/6] Geração ativada (--generate-datataset): Gerando novos dados sintéticos...")
        
        df, data_path = generate_and_save(N_SAMPLES)
        data_filename = os.path.basename(data_path)
        print(f"  Novo dataset criado em: {data_path} ({len(df)} amostras)")
    else:
        print("\n[1/6] Carregando conjunto de dados existente...")
        df = pd.read_csv(fixed_path)
        data_path = fixed_path
        data_filename = os.path.basename(fixed_path)
        
        # Obter o timestamp do dataset gerado mais recentemente se possível para rotulagem coerente
        try:
            import glob
            files = [f for f in glob.glob("data/dados_academicos_*.csv") if os.path.isfile(f)]
            if files:
                latest = max(files, key=os.path.getmtime)
                data_filename = os.path.basename(latest)
        except Exception:
            pass
            
        print(f"  Dados carregados com sucesso de: {fixed_path} ({len(df)} amostras)")
        
    print_distribution_check(df)

    # 2. Treinar Árvore de Decisão
    print("\n[2/6] Treinando Árvore de Decisão (com GridSearchCV)...")
    dt_results = run_decision_tree(df)
    print(f"  Holdout: Acc={dt_results['holdout']['accuracy']:.4f} | F1={dt_results['holdout']['f1_weighted']:.4f}")
    print(f"  CV k=10: Acc={dt_results['cv']['accuracy']['test_mean']:.4f} (±{dt_results['cv']['accuracy']['test_std']:.4f})")

    # 3. Treinar Floresta Aleatória
    print("\n[3/6] Treinando Floresta Aleatória (com GridSearchCV)...")
    rf_results = run_random_forest(df)
    print(f"  Holdout: Acc={rf_results['holdout']['accuracy']:.4f} | F1={rf_results['holdout']['f1_weighted']:.4f}")
    print(f"  CV k=10: Acc={rf_results['cv']['accuracy']['test_mean']:.4f} (±{rf_results['cv']['accuracy']['test_std']:.4f})")

    # 4. Treinar Gradient Boosting (XGBoost)
    print("\n[4/6] Treinando Gradient Boosting (com GridSearchCV)...")
    gb_results = run_gradient_boosting(df)
    print(f"  Holdout: Acc={gb_results['holdout']['accuracy']:.4f} | F1={gb_results['holdout']['f1_weighted']:.4f}")
    print(f"  CV k=10: Acc={gb_results['cv']['accuracy']['test_mean']:.4f} (±{gb_results['cv']['accuracy']['test_std']:.4f})")

    # Baselines de referência (Dummy majoritário + Regressão Logística)
    print("\n[-] Avaliando baselines (Dummy majoritário e Regressão Logística)...")
    baseline_results = run_baselines(df)
    for name, res in baseline_results.items():
        print(f"  {name}: Acc={res['holdout']['accuracy']:.4f} | F1={res['holdout']['f1_weighted']:.4f}")

    # Teste de Wilcoxon pareado sobre os folds (XGB × RF × DT)
    print("\n[-] Teste de Wilcoxon pareado sobre os folds da CV (F1 ponderado)...")
    wilcoxon_tests({
        "Árvore de Decisão": dt_results,
        "Floresta Aleatória": rf_results,
        gb_results["model_name"]: gb_results,
    })

    # Gerar gráficos comparativos salvando na pasta de imagens do LaTeX
    print("\n[-] Gerando gráficos de comparação (ROC, Matrizes, Importâncias)...")
    save_comparison_plots(dt_results, rf_results, gb_results, dt_results["le_target"])

    # 5. Executar notebooks e salvar com timestamp
    timestamp = data_filename.replace("dados_academicos_", "").replace(".csv", "")
    print("\n[5/6] Executando notebooks...")
    execute_notebooks(timestamp)

    # 6. Salvar relatório de métricas
    print("\n[6/6] Salvando relatório de métricas...")
    env_vars = get_env_variables_display()
    output_content = build_output_md(
        dt_results, rf_results, gb_results,
        data_filename, env_vars, baseline_results,
    )

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"result-{data_filename.replace('.csv', '')}.md"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    # Etapa opcional de pós-análise (extensões locais, não versionadas)
    try:
        from local.post_analysis import run as post_analysis
        print("\n[-] Executando pós-análise (extensão local)...")
        post_analysis({
            "dt_results": dt_results,
            "rf_results": rf_results,
            "gb_results": gb_results,
            "baseline_results": baseline_results,
            "data_filename": data_filename,
            "n_samples": len(df),
            "output_path": output_path,
        })
    except ImportError:
        pass

    print(f"\n{'=' * 60}")
    print(f"Relatório salvo em: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
