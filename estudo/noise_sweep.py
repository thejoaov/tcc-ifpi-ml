"""
Varredura de NOISE_LEVEL (B6) - caracterização do problema.

O NOISE_LEVEL controla a variância IDIOSSINCRÁTICA da deriva do 2º semestre
(fatores não observáveis: saúde, eventos familiares, motivação):
  - sigma da média final projetada = NOISE_LEVEL / 10 (pontos de 0-10)
  - sigma da frequência projetada  = NOISE_LEVEL * 0,8 (p.p.)

A varredura NÃO busca uma acurácia-alvo: ela caracteriza como a
previsibilidade do desfecho anual degrada conforme cresce a variância não
observável. O valor adotado no estudo é uma decisão fenomenológica
(registrada na matriz de rastreabilidade), e a curva é reportada como
análise de sensibilidade.

Uso: .venv/bin/python noise_sweep.py
Saída: data/varredura_ruido.md|.csv
"""

import os
import warnings

warnings.filterwarnings("ignore")

import importlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import generate_data
import models

LEVELS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
K_FOLDS = 5  # varredura usa k=5 por custo; o estudo principal usa k=10


def evaluate_level(level: float, random_state: int = 42) -> dict:
    os.environ["NOISE_LEVEL"] = str(level)
    importlib.reload(generate_data)
    df = generate_data.generate_synthetic_data(generate_data.N_SAMPLES)

    X, y, _, _ = models.preprocess(df)
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", RandomForestClassifier(
            random_state=random_state, n_jobs=-1, n_estimators=200, max_depth=15
        )),
    ])
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)
    res = cross_validate(
        pipeline, X, y, cv=cv,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
    )

    classes, counts = np.unique(df["situacao"], return_counts=True)
    dist = {c: n / len(df) * 100 for c, n in zip(classes, counts)}
    return {
        "noise_level": level,
        "sigma_media": level / 10.0,
        "sigma_freq_pp": level * 0.8,
        "acuracia_cv": res["test_accuracy"].mean(),
        "acuracia_std": res["test_accuracy"].std(),
        "f1_cv": res["test_f1_weighted"].mean(),
        "f1_std": res["test_f1_weighted"].std(),
        "pct_aprovado": dist.get("Aprovado", 0.0),
        "pct_em_risco": dist.get("Em Risco", 0.0),
        "pct_reprovado": dist.get("Reprovado", 0.0),
    }


def main():
    original = os.environ.get("NOISE_LEVEL")
    rows = []
    for level in LEVELS:
        r = evaluate_level(level)
        rows.append(r)
        print(f"  NOISE_LEVEL={level:4.1f} (sigma_media={r['sigma_media']:.2f}) | "
              f"Acc={r['acuracia_cv']:.4f} | F1={r['f1_cv']:.4f} | "
              f"classes={r['pct_aprovado']:.0f}/{r['pct_em_risco']:.0f}/{r['pct_reprovado']:.0f}")

    # Restaurar o ambiente
    if original is not None:
        os.environ["NOISE_LEVEL"] = original
    else:
        os.environ.pop("NOISE_LEVEL", None)
    importlib.reload(generate_data)

    result = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    result.to_csv(os.path.join("data", "varredura_ruido.csv"), index=False)
    with open(os.path.join("data", "varredura_ruido.md"), "w", encoding="utf-8") as f:
        f.write("# Varredura de NOISE_LEVEL - sensibilidade ao ruído idiossincrático\n\n")
        f.write("Floresta Aleatória (parâmetros fixos), CV estratificada k=5 com SMOTE no fold.\n")
        f.write("sigma_media = NOISE_LEVEL/10 pontos (escala 0-10); "
                "sigma_freq = NOISE_LEVEL*0,8 p.p.\n\n")
        f.write("| NOISE_LEVEL | σ média | σ freq (p.p.) | Acurácia (CV) | F1 (CV) | Aprovado/Em Risco/Reprovado (%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for _, r in result.iterrows():
            f.write(
                f"| {r['noise_level']:.1f} | {r['sigma_media']:.2f} | {r['sigma_freq_pp']:.1f} "
                f"| {r['acuracia_cv']:.4f} ± {r['acuracia_std']:.4f} "
                f"| {r['f1_cv']:.4f} ± {r['f1_std']:.4f} "
                f"| {r['pct_aprovado']:.1f} / {r['pct_em_risco']:.1f} / {r['pct_reprovado']:.1f} |\n"
            )
    print("  Tabela exportada em data/varredura_ruido.md|.csv")


if __name__ == "__main__":
    main()
