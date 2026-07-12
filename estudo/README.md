# Estudo Comparativo de Modelos Preditivos - Desempenho Acadêmico

Estudo do TCC da Licenciatura em Informática (IFPI Zona Sul) que compara três modelos de classificação para predição da tendência de desempenho acadêmico de estudantes do ensino médio da rede pública estadual do Piauí, a partir de um conjunto de dados sintético parametrizado em estatísticas oficiais.

## Modelos

| Modelo             | Algoritmo                                      | Notebook                         |
| ------------------ | ---------------------------------------------- | -------------------------------- |
| Árvore de Decisão  | `DecisionTreeClassifier`                       | `models/decision_tree.ipynb`     |
| Floresta Aleatória | `RandomForestClassifier`                       | `models/random_forest.ipynb`     |
| Gradient Boosting  | `XGBClassifier` / `GradientBoostingClassifier` | `models/gradient_boosting.ipynb` |

Baselines de referência: `DummyClassifier` (classe majoritária) e Regressão Logística multinomial.

## Etapas metodológicas

1. **Geração de dados sintéticos** - configurável via `.env`; parâmetros rastreados em `data/parametros_fontes.md` e auditados em `data/auditoria_fontes.md`
2. **Pré-processamento** - One-Hot Encoding (nominais), Label Encoding (binárias), Feature Engineering (variáveis derivadas), remoção de features protegidas/de baixa importância
3. **Balanceamento** - SMOTE aplicado dentro de cada fold (sem data leakage)
4. **Otimização** - GridSearchCV (5-fold) para seleção de hiperparâmetros
5. **Treinamento e avaliação** - holdout (70/30) + validação cruzada (k=10, Stratified K-Fold); baselines e teste de Wilcoxon pareado sobre os folds
6. **Relatório de métricas** - salvo em `output/result-*.md`

## Estrutura

```
estudo/
├── run.sh                  # Script principal (executa tudo)
├── run.py                  # Orquestrador Python
├── generate_data.py        # Geração de dados sintéticos + matriz de rastreabilidade
├── models.py               # Funções compartilhadas (preprocess, split, evaluate, baselines, ablation, Wilcoxon)
├── noise_sweep.py          # Varredura de sensibilidade ao ruído idiossincrático
├── models/
│   ├── decision_tree.ipynb
│   ├── random_forest.ipynb
│   └── gradient_boosting.ipynb
├── data/                   # CSVs gerados + matriz de rastreabilidade + tabelas exportadas
├── output/                 # Relatórios .md gerados
├── .env                    # Configuração padrão (rede pública estadual do PI)
├── .env.nacional           # Contraste: turma média nacional (BR)
├── .env.publica            # Contraste: alta vulnerabilidade (interior/zona rural)
├── .env.elite              # Contraste: rede privada de alto nível socioeconômico
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=tcc-estudo
```

## Configuração

Copie o `.env.example` para `.env` e ajuste se necessário:

```bash
cp .env.example .env
```

Variáveis principais:

| Variável                    | Descrição                                      | Padrão    |
| --------------------------- | ---------------------------------------------- | --------- |
| `N_SAMPLES`                 | Número de amostras sintéticas                  | 3000      |
| `SEED`                      | Seed para reprodutibilidade                    | 42        |
| `LDB_MEDIA_MINIMA`          | Média mínima de aprovação (LDB)                | 6.0       |
| `LDB_FREQ_MINIMA`           | Frequência mínima de aprovação (LDB, %)        | 75        |
| `BANDA_RISCO_MEDIA_INF/SUP` | Banda pedagógica "Em Risco" (média)            | 5.0 / 6.5 |
| `BANDA_RISCO_FREQ_SUP`      | Banda pedagógica "Em Risco" (frequência)       | 80        |
| `NOISE_LEVEL`               | Ruído idiossincrático da deriva do 2º semestre | 3.0       |

As distribuições demográficas, socioeconômicas e acadêmicas também são configuráveis via `.env`. Veja `.env.example` para todas as opções.

## Uso

### Execução padrão (usa `.env`)

```bash
./run.sh
```

### Execução com cenário de contraste

```bash
./run.sh --env .env.nacional   # Turma média nacional
./run.sh --env .env.publica    # Alta vulnerabilidade
./run.sh --env .env.elite      # Rede privada de elite
```

### O que acontece ao executar

1. Carrega (ou gera com `--generate-dataset`) o dataset sintético em `data/`
2. Treina os 3 modelos com GridSearchCV + validação cruzada
3. Avalia os baselines e executa o teste de Wilcoxon pareado sobre os folds
4. Gera os gráficos comparativos (ROC, matrizes de confusão, importâncias)
5. Executa os 3 notebooks e salva cópias com timestamp em `models/`
6. Salva o relatório de métricas em `output/result-*.md`

### Análises complementares

```bash
.venv/bin/python noise_sweep.py   # Varredura de NOISE_LEVEL -> data/varredura_ruido.md
```

O estudo de ablation das features derivadas está em `models.run_ablation_study` (exporta `data/ablation_features.md`).

## Variável alvo e semântica temporal

As variáveis acadêmicas observáveis (`nota_media`, `frequencia`, `media_atividades`, `taxa_entrega_atividades`) são **parciais do 1º semestre**. O alvo `situacao` é a **tendência ao final do ano letivo**, projetada como sinal parcial + deriva do 2º semestre + ruído idiossincrático, e rotulada pelos critérios da LDB (Lei nº 9.394/1996, art. 24):

- **Aprovado** - média final projetada ≥ 6,5 e frequência projetada ≥ 80% (fora da zona de fronteira)
- **Em Risco** - média projetada em [5,0; 6,5) ou frequência projetada em [75; 80) (banda pedagógica fixa)
- **Reprovado** - média projetada < 5,0 ou frequência projetada < 75%

As proporções de classe são emergentes (não calibradas).

## Dependências

- Python 3.10+
- scikit-learn, pandas, numpy, scipy, matplotlib, seaborn
- imbalanced-learn (SMOTE)
- xgboost
- python-dotenv
- jupyter (nbconvert para execução dos notebooks)
