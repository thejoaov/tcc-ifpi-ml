# Estudo de ablation - features derivadas

CV estratificada k=10 com SMOTE no fold; hiperparâmetros fixos moderados.
`score_academico` foi removido do conjunto final por proximidade estrutural com o rótulo (combinação linear de nota/frequência/reprovações).

| Configuração                           | Modelo             | Nº features | Acurácia (CV)   | F1 (CV)         |
| -------------------------------------- | ------------------ | ----------- | --------------- | --------------- |
| sem derivadas                          | Árvore de Decisão  | 16          | 0.8007 ± 0.0231 | 0.8010 ± 0.0228 |
| sem derivadas                          | Floresta Aleatória | 16          | 0.8487 ± 0.0235 | 0.8493 ± 0.0228 |
| sem derivadas                          | XGBoost            | 16          | 0.8450 ± 0.0193 | 0.8449 ± 0.0192 |
| + nota_x_frequencia                    | Árvore de Decisão  | 17          | 0.8087 ± 0.0207 | 0.8094 ± 0.0200 |
| + nota_x_frequencia                    | Floresta Aleatória | 17          | 0.8470 ± 0.0202 | 0.8470 ± 0.0196 |
| + nota_x_frequencia                    | XGBoost            | 17          | 0.8473 ± 0.0165 | 0.8469 ± 0.0166 |
| + reprov_por_serie                     | Árvore de Decisão  | 17          | 0.8033 ± 0.0202 | 0.8047 ± 0.0190 |
| + reprov_por_serie                     | Floresta Aleatória | 17          | 0.8473 ± 0.0185 | 0.8480 ± 0.0181 |
| + reprov_por_serie                     | XGBoost            | 17          | 0.8450 ± 0.0202 | 0.8449 ± 0.0200 |
| + score_academico                      | Árvore de Decisão  | 17          | 0.7960 ± 0.0222 | 0.7971 ± 0.0218 |
| + score_academico                      | Floresta Aleatória | 17          | 0.8420 ± 0.0206 | 0.8425 ± 0.0197 |
| + score_academico                      | XGBoost            | 17          | 0.8410 ± 0.0227 | 0.8408 ± 0.0222 |
| final (nota_x_freq + reprov_por_serie) | Árvore de Decisão  | 18          | 0.7970 ± 0.0169 | 0.7982 ± 0.0153 |
| final (nota_x_freq + reprov_por_serie) | Floresta Aleatória | 18          | 0.8463 ± 0.0213 | 0.8465 ± 0.0207 |
| final (nota_x_freq + reprov_por_serie) | XGBoost            | 18          | 0.8493 ± 0.0221 | 0.8490 ± 0.0223 |
| final + score_academico                | Árvore de Decisão  | 19          | 0.8020 ± 0.0216 | 0.8028 ± 0.0210 |
| final + score_academico                | Floresta Aleatória | 19          | 0.8453 ± 0.0218 | 0.8453 ± 0.0212 |
| final + score_academico                | XGBoost            | 19          | 0.8490 ± 0.0209 | 0.8487 ± 0.0205 |
