"""
Módulo com funções reutilizáveis para treino e avaliação dos modelos.
Usado pelo orquestrador (run.py) e pode ser importado pelos notebooks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# --- Variáveis nominais (sem ordem) vs binárias ---
NOMINAL_COLS = ["estado_civil", "renda_per_capita", "cor_raca"]
BINARY_COLS = [
    "genero", "trabalha", "acesso_internet", "possui_computador",
    "tipo_escola_origem", "distorcao_idade_serie", "assistencia_pe_de_meia",
]

# --- Features removidas do treino ---
# genero e cor_raca: proteção contra viés discriminatório (peso zero no
# fenômeno gerador; usadas apenas para caracterização da turma).
# tipo_escola_origem: baixa importância observada.
LOW_IMPORTANCE_FEATURES = [
    "genero", "cor_raca", "tipo_escola_origem",
]


# Features derivadas disponíveis (para o estudo de ablation).
# `score_academico` foi REMOVIDO do conjunto final: sua construção
# (combinação linear de nota/frequência/reprovações) é estruturalmente
# próxima do rótulo, inflando métricas - ver estudo de ablation.
DERIVED_FEATURES = ["nota_x_frequencia", "reprov_por_serie"]


def feature_engineering(df: pd.DataFrame, derived: list[str] | None = None) -> pd.DataFrame:
    """Cria variáveis derivadas para melhorar a capacidade preditiva.

    `derived` permite selecionar quais features derivadas criar
    (default: DERIVED_FEATURES - sem `score_academico`)."""
    if derived is None:
        derived = DERIVED_FEATURES
    df = df.copy()
    if "nota_x_frequencia" in derived:
        df["nota_x_frequencia"] = df["nota_media"] * df["frequencia"] / 100.0
    if "reprov_por_serie" in derived:
        df["reprov_por_serie"] = df["reprovacoes"] / df["serie"].clip(lower=1)
    if "score_academico" in derived:
        # Mantida apenas para o estudo de ablation (não usada no conjunto final)
        df["score_academico"] = (
            df["nota_media"] * 0.4 + df["frequencia"] * 0.04
            - df["reprovacoes"] * 1.2
        )
    return df


def preprocess(
    df: pd.DataFrame,
    remove_low_importance: bool = True,
    derived: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict, LabelEncoder]:
    """Codifica variáveis categóricas com One-Hot Encoding para nominais
    e Label Encoding para binárias. Aplica feature engineering.
    `derived` seleciona as features derivadas (ver feature_engineering).
    Retorna (X, y, encoders_info, le_target)."""
    df_processed = df.copy()

    # Separar target antes de tudo
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_processed["situacao"])
    df_processed = df_processed.drop("situacao", axis=1)

    # Feature Engineering
    df_processed = feature_engineering(df_processed, derived=derived)

    # Remover features de baixa importância/protegidas ANTES da codificação
    if remove_low_importance:
        cols_to_drop = [c for c in LOW_IMPORTANCE_FEATURES if c in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop, errors="ignore")

    # Label Encoding para binárias (Sim/Nao -> 1/0)
    label_encoders = {}
    for col in BINARY_COLS:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le

    # One-Hot Encoding para nominais (sem ordem)
    nominal_present = [c for c in NOMINAL_COLS if c in df_processed.columns]
    df_processed = pd.get_dummies(df_processed, columns=nominal_present, drop_first=True)

    X = df_processed
    encoders_info = {"label_encoders": label_encoders, "nominal_cols": NOMINAL_COLS}

    return X, y, encoders_info, le_target


def split_and_balance(X, y, test_size=0.3, random_state=42):
    """Split treino/teste e aplica SMOTE no treino."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_bal, y_train_bal


def evaluate_model(model, X_test, y_test, le_target) -> dict:
    """Avalia modelo no conjunto de teste. Retorna dict de métricas."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    target_names = le_target.classes_

    # Calcular AUC multiclasse One-vs-Rest (OvR) weighted
    try:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc_score = 0.0

    report_str = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "auc_weighted": auc_score,
        "classification_report": report_str,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_test,
    }


def save_comparison_plots(dt_results, rf_results, gb_results, le_target):
    """Gera e salva na pasta de imagens do LaTeX os gráficos comparativos de ROC, Matriz de Confusão e Importância de Atributos."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGES_DIR = os.path.join(BASE_DIR, "documento", "imagens")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    classes = le_target.classes_
    n_classes = len(classes)
    models_data = [dt_results, rf_results, gb_results]

    # Configuração de estilo geral para artigos
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })

    # 1. CURVAS ROC (OvR)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, res in enumerate(models_data):
        ax = axes[idx]
        y_test = res["holdout"]["y_true"]
        y_prob = res["holdout"]["y_prob"]
        
        # Binarizar para cálculo OvR ROC
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Plotar curva para cada classe
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.2f})")
            
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taxa de Falso Positivo")
        ax.set_ylabel("Taxa de Verdadeiro Positivo")
        ax.set_title(f"{res['model_name']}")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Curvas ROC Multiclasse One-vs-Rest (OvR) - Comparação de Modelos", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "roc_curves_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. MATRIZES DE CONFUSÃO
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, res in enumerate(models_data):
        ax = axes[idx]
        y_test = res["holdout"]["y_true"]
        y_pred = res["holdout"]["y_pred"]
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 13, "weight": "bold"},
            cbar=False
        )
        ax.set_xlabel("Classe Predita", fontsize=12)
        ax.set_ylabel("Classe Real", fontsize=12)
        ax.set_title(f"{res['model_name']}")

    plt.suptitle("Matrizes de Confusão Comparativas", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrices_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. IMPORTÂNCIA DOS ATRIBUTOS (Top 8 por modelo)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, res in enumerate(models_data):
        ax = axes[idx]
        fi = res["feature_importance"].head(8)
        
        # Mapeamento do nome dos atributos sintéticos para nomes legíveis
        names_map = {
            "idade": "Idade",
            "serie": "Série (EM)",
            "nota_media": "Nota Média (1º sem.)",
            "frequencia": "Frequência % (1º sem.)",
            "media_atividades": "Média de Atividades (1º sem.)",
            "taxa_entrega_atividades": "Taxa de Entrega de Atividades",
            "reprovacoes": "Reprovações Acumuladas",
            "nota_x_frequencia": "Nota × Frequência",
            "reprov_por_serie": "Reprovações por Série",
            "score_academico": "Score Acadêmico Composto",
            "acesso_internet": "Acesso à Internet",
            "possui_computador": "Possui Computador",
            "trabalha": "Estudante-Trabalhador",
            "distorcao_idade_serie": "Distorção Idade-Série",
            "assistencia_pe_de_meia": "Pé-de-Meia",
            "renda_per_capita_0,5-1 SM": "Renda p.c. 0,5-1 SM",
            "renda_per_capita_Acima de 1 SM": "Renda p.c. > 1 SM",
            "estado_civil_Casado/Uniao": "Civil: Casado/União",
            "estado_civil_Outro": "Civil: Outro",
        }
        fi = fi.copy()
        fi["atributo_exibicao"] = fi["atributo"].map(lambda x: names_map.get(x, x))
        
        sns.barplot(
            x="importancia", y="atributo_exibicao", data=fi, ax=ax,
            palette="viridis", hue="atributo_exibicao", legend=False
        )
        ax.set_xlabel("Importância Relativa")
        ax.set_ylabel("")
        ax.set_title(f"{res['model_name']}")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Importância dos Atributos mais Relevantes", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importances_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  [OK] Gráficos comparativos salvos com sucesso em: {IMAGES_DIR}")


def cross_validate_pipeline(estimator_name: str, estimator, X_train, y_train, k=10, random_state=42) -> dict:
    """Executa cross-validation com SMOTE dentro de cada fold."""
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", estimator),
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
    )

    results = {}
    for metric in scoring:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        results[metric] = {
            "test_mean": test_scores.mean(),
            "test_std": test_scores.std(),
            "train_mean": train_scores.mean(),
            "train_std": train_scores.std(),
            "test_per_fold": test_scores.tolist(),
        }

    return results


def tune_hyperparameters(estimator, param_grid, X_train, y_train, random_state=42):
    """Executa GridSearchCV com SMOTE dentro de cada fold para encontrar melhores hiperparâmetros."""
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", estimator),
    ])

    # Prefixar params com 'classifier__'
    prefixed_grid = {f"classifier__{k}": v for k, v in param_grid.items()}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        prefixed_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    print(f"  Melhores hiperparâmetros: {best_params}")
    print(f"  Melhor F1 (CV): {grid_search.best_score_:.4f}")

    return best_params


def run_baselines(df: pd.DataFrame, k: int = 10, random_state: int = 42) -> dict:
    """Baselines de referência (B5/D9):

    - DummyClassifier (classe majoritária), SEM SMOTE - piso de comparação
      sobre a distribuição original de classes.
    - Regressão Logística multinomial, com o MESMO pré-processamento dos
      demais modelos (One-Hot, feature engineering) + StandardScaler e
      SMOTE dentro de cada fold.

    Retorna dict {nome: resultados} no mesmo formato dos run_* principais.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, _, le_target = preprocess(df)
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = split_and_balance(
        X, y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    results = {}

    # --- Dummy (majoritário), sem SMOTE ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_holdout = evaluate_model(dummy, X_test, y_test, le_target)
    dummy_cv_raw = cross_validate(
        DummyClassifier(strategy="most_frequent"), X_train, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
    )
    results["Baseline Majoritário (Dummy)"] = {
        "model_name": "Baseline Majoritário (Dummy)",
        "holdout": dummy_holdout,
        "cv": _summarize_cv(dummy_cv_raw, scoring),
        "le_target": le_target,
    }

    # --- Regressão Logística multinomial (scaler + SMOTE no fold) ---
    def make_lr_pipeline():
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ])

    lr_pipe = make_lr_pipeline()
    lr_pipe.fit(X_train, y_train)
    lr_holdout = evaluate_model(lr_pipe, X_test, y_test, le_target)
    lr_cv_raw = cross_validate(
        make_lr_pipeline(), X_train, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
    )
    results["Regressão Logística"] = {
        "model_name": "Regressão Logística",
        "holdout": lr_holdout,
        "cv": _summarize_cv(lr_cv_raw, scoring),
        "le_target": le_target,
    }

    return results


def _summarize_cv(cv_results: dict, scoring: dict) -> dict:
    """Converte a saída bruta do cross_validate no formato usado no estudo."""
    out = {}
    for metric in scoring:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]
        out[metric] = {
            "test_mean": test_scores.mean(),
            "test_std": test_scores.std(),
            "train_mean": train_scores.mean(),
            "train_std": train_scores.std(),
            "test_per_fold": test_scores.tolist(),
        }
    return out


def wilcoxon_tests(results_by_name: dict, metric: str = "f1_weighted", output_dir: str = "data") -> pd.DataFrame:
    """Teste de Wilcoxon pareado (B5/D9) sobre os folds da CV k=10.

    `results_by_name`: {nome: resultados run_*} - os folds são pareáveis
    porque todos usam a MESMA StratifiedKFold (shuffle, random_state=42).
    Exporta data/baselines_wilcoxon.md|.csv e retorna o DataFrame.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon

    rows = []
    for (name_a, res_a), (name_b, res_b) in combinations(results_by_name.items(), 2):
        folds_a = np.array(res_a["cv"][metric]["test_per_fold"])
        folds_b = np.array(res_b["cv"][metric]["test_per_fold"])
        diff = folds_a - folds_b
        if np.allclose(diff, 0):
            stat, p_value = np.nan, 1.0
        else:
            stat, p_value = wilcoxon(folds_a, folds_b)
        rows.append({
            "comparacao": f"{name_a} × {name_b}",
            "metrica": metric,
            f"media_{metric}_A": folds_a.mean(),
            f"media_{metric}_B": folds_b.mean(),
            "estatistica_W": stat,
            "p_valor": p_value,
            "significativo_5pct": "sim" if p_value < 0.05 else "não",
        })
        print(f"  [wilcoxon] {rows[-1]['comparacao']:60s} p={p_value:.4f} "
              f"({'significativo' if p_value < 0.05 else 'equivalente'})")

    result = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, "baselines_wilcoxon.csv"), index=False)
    with open(os.path.join(output_dir, "baselines_wilcoxon.md"), "w", encoding="utf-8") as f:
        f.write("# Teste de Wilcoxon pareado sobre os folds (CV k=10)\n\n")
        f.write(f"Métrica: {metric}. Folds pareados (mesma StratifiedKFold, seed 42).\n\n")
        f.write("| Comparação | Média A | Média B | W | p-valor | Significativo (5%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for _, r in result.iterrows():
            w_str = "-" if pd.isna(r["estatistica_W"]) else f"{r['estatistica_W']:.1f}"
            f.write(
                f"| {r['comparacao']} | {r[f'media_{metric}_A']:.4f} "
                f"| {r[f'media_{metric}_B']:.4f} | {w_str} "
                f"| {r['p_valor']:.4f} | {r['significativo_5pct']} |\n"
            )
    print(f"  [wilcoxon] Tabela exportada em {output_dir}/baselines_wilcoxon.md|.csv")
    return result


def run_ablation_study(df: pd.DataFrame, output_dir: str = "data", k: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Estudo de ablation das features derivadas (B4/D4).

    Treina DT/RF/XGB (hiperparâmetros fixos moderados, SMOTE no fold,
    CV estratificada k=10) sob configurações: sem derivadas, com cada
    derivada isolada, conjunto final (sem score_academico) e conjunto
    final + score_academico. Exporta tabela em MD/CSV e retorna o DataFrame.
    """
    configs = {
        "sem derivadas": [],
        "+ nota_x_frequencia": ["nota_x_frequencia"],
        "+ reprov_por_serie": ["reprov_por_serie"],
        "+ score_academico": ["score_academico"],
        "final (nota_x_freq + reprov_por_serie)": DERIVED_FEATURES,
        "final + score_academico": DERIVED_FEATURES + ["score_academico"],
    }

    estimators = {
        "Árvore de Decisão": DecisionTreeClassifier(
            random_state=random_state, max_depth=10, min_samples_leaf=3
        ),
        "Floresta Aleatória": RandomForestClassifier(
            random_state=random_state, n_jobs=-1, n_estimators=200, max_depth=15
        ),
    }
    if HAS_XGBOOST:
        estimators["XGBoost"] = XGBClassifier(
            random_state=random_state, n_jobs=-1, eval_metric="mlogloss",
            n_estimators=200, max_depth=5, learning_rate=0.1,
        )
    else:
        estimators["Gradient Boosting"] = GradientBoostingClassifier(
            random_state=random_state, n_estimators=200, max_depth=5, learning_rate=0.1
        )

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    rows = []
    for config_name, derived in configs.items():
        X, y, _, _ = preprocess(df, derived=derived)
        for model_name, est in estimators.items():
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", est),
            ])
            cv_res = cross_validate(
                pipeline, X, y, cv=cv,
                scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
            )
            rows.append({
                "configuracao": config_name,
                "modelo": model_name,
                "n_features": X.shape[1],
                "acuracia_cv": cv_res["test_accuracy"].mean(),
                "acuracia_std": cv_res["test_accuracy"].std(),
                "f1_cv": cv_res["test_f1_weighted"].mean(),
                "f1_std": cv_res["test_f1_weighted"].std(),
            })
            print(f"  [ablation] {config_name:42s} | {model_name:18s} | "
                  f"Acc={rows[-1]['acuracia_cv']:.4f} | F1={rows[-1]['f1_cv']:.4f}")

    result = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, "ablation_features.csv"), index=False)
    with open(os.path.join(output_dir, "ablation_features.md"), "w", encoding="utf-8") as f:
        f.write("# Estudo de ablation - features derivadas\n\n")
        f.write("CV estratificada k=10 com SMOTE no fold; hiperparâmetros fixos moderados.\n")
        f.write("`score_academico` foi removido do conjunto final por proximidade "
                "estrutural com o rótulo (combinação linear de nota/frequência/reprovações).\n\n")
        f.write("| Configuração | Modelo | Nº features | Acurácia (CV) | F1 (CV) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for _, r in result.iterrows():
            f.write(
                f"| {r['configuracao']} | {r['modelo']} | {r['n_features']} "
                f"| {r['acuracia_cv']:.4f} ± {r['acuracia_std']:.4f} "
                f"| {r['f1_cv']:.4f} ± {r['f1_std']:.4f} |\n"
            )
    print(f"  [ablation] Tabela exportada em {output_dir}/ablation_features.md|.csv")
    return result


def run_decision_tree(df: pd.DataFrame) -> dict:
    """Executa pipeline completo de Árvore de Decisão com tuning. Retorna resultados."""
    X, y, encoders_info, le_target = preprocess(df)
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = split_and_balance(X, y)

    # GridSearchCV para encontrar melhores hiperparâmetros
    param_grid = {
        "max_depth": [5, 8, 10, 12, 15],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [2, 3, 5, 8],
        "criterion": ["gini", "entropy"],
    }

    best_params = tune_hyperparameters(
        DecisionTreeClassifier(random_state=42),
        param_grid, X_train, y_train
    )

    # Treinar modelo com melhores hiperparâmetros
    dt_model = DecisionTreeClassifier(random_state=42, **best_params)
    dt_model.fit(X_train_bal, y_train_bal)

    # Avaliação holdout
    holdout_metrics = evaluate_model(dt_model, X_test, y_test, le_target)

    # Cross-validation com melhores hiperparâmetros
    cv_results = cross_validate_pipeline(
        "DecisionTree",
        DecisionTreeClassifier(random_state=42, **best_params),
        X_train, y_train
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        "atributo": X.columns,
        "importancia": dt_model.feature_importances_
    }).sort_values("importancia", ascending=False)

    return {
        "model_name": "Árvore de Decisão",
        "holdout": holdout_metrics,
        "cv": cv_results,
        "feature_importance": feature_importance,
        "n_train": X_train.shape[0],
        "n_train_balanced": X_train_bal.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X.shape[1],
        "params": best_params,
        "le_target": le_target,
    }


def run_random_forest(df: pd.DataFrame) -> dict:
    """Executa pipeline completo de Floresta Aleatória com tuning. Retorna resultados."""
    X, y, encoders_info, le_target = preprocess(df)
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = split_and_balance(X, y)

    # GridSearchCV para encontrar melhores hiperparâmetros
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"],
    }

    best_params = tune_hyperparameters(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, X_train, y_train
    )

    # Treinar modelo com melhores hiperparâmetros
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    rf_model.fit(X_train_bal, y_train_bal)

    # Avaliação holdout
    holdout_metrics = evaluate_model(rf_model, X_test, y_test, le_target)

    # Cross-validation com melhores hiperparâmetros
    cv_results = cross_validate_pipeline(
        "RandomForest",
        RandomForestClassifier(random_state=42, n_jobs=-1, **best_params),
        X_train, y_train
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        "atributo": X.columns,
        "importancia": rf_model.feature_importances_
    }).sort_values("importancia", ascending=False)

    return {
        "model_name": "Floresta Aleatória",
        "holdout": holdout_metrics,
        "cv": cv_results,
        "feature_importance": feature_importance,
        "n_train": X_train.shape[0],
        "n_train_balanced": X_train_bal.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X.shape[1],
        "params": best_params,
        "le_target": le_target,
    }


def run_gradient_boosting(df: pd.DataFrame) -> dict:
    """Executa pipeline completo de Gradient Boosting (XGBoost) com tuning. Retorna resultados."""
    X, y, encoders_info, le_target = preprocess(df)
    X_train, X_test, y_train, y_test, X_train_bal, y_train_bal = split_and_balance(X, y)

    if HAS_XGBOOST:
        # GridSearchCV com XGBoost
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        best_params = tune_hyperparameters(
            XGBClassifier(
                random_state=42, n_jobs=-1, eval_metric="mlogloss",
            ),
            param_grid, X_train, y_train
        )

        # Treinar modelo com melhores hiperparâmetros
        gb_model = XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric="mlogloss",
            **best_params
        )
        gb_model.fit(X_train_bal, y_train_bal)

        # Cross-validation
        cv_results = cross_validate_pipeline(
            "XGBoost",
            XGBClassifier(
                random_state=42, n_jobs=-1, eval_metric="mlogloss",
                **best_params
            ),
            X_train, y_train
        )
    else:
        # Fallback: sklearn GradientBoostingClassifier
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        }

        best_params = tune_hyperparameters(
            GradientBoostingClassifier(random_state=42),
            param_grid, X_train, y_train
        )

        # Treinar modelo com melhores hiperparâmetros
        gb_model = GradientBoostingClassifier(random_state=42, **best_params)
        gb_model.fit(X_train_bal, y_train_bal)

        # Cross-validation
        cv_results = cross_validate_pipeline(
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42, **best_params),
            X_train, y_train
        )

    # Avaliação holdout
    holdout_metrics = evaluate_model(gb_model, X_test, y_test, le_target)

    # Feature importance
    feature_importance = pd.DataFrame({
        "atributo": X.columns,
        "importancia": gb_model.feature_importances_
    }).sort_values("importancia", ascending=False)

    model_name = "XGBoost" if HAS_XGBOOST else "Gradient Boosting"

    return {
        "model_name": model_name,
        "holdout": holdout_metrics,
        "cv": cv_results,
        "feature_importance": feature_importance,
        "n_train": X_train.shape[0],
        "n_train_balanced": X_train_bal.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X.shape[1],
        "params": best_params,
        "le_target": le_target,
    }
