# Teste de Wilcoxon pareado sobre os folds (CV k=10)

Métrica: f1_weighted. Folds pareados (mesma StratifiedKFold, seed 42).

| Comparação | Média A | Média B | W | p-valor | Significativo (5%) |
| --- | --- | --- | --- | --- | --- |
| Árvore de Decisão × Floresta Aleatória | 0.8050 | 0.8503 | 0.0 | 0.0020 | sim |
| Árvore de Decisão × XGBoost | 0.8050 | 0.8554 | 0.0 | 0.0020 | sim |
| Floresta Aleatória × XGBoost | 0.8503 | 0.8554 | 15.0 | 0.2324 | não |
