# TCC - Modelo Preditivo para Avaliação de Desempenho Acadêmico

Estudo comparativo entre **Árvore de Decisão**, **Floresta Aleatória** e **Gradient Boosting (XGBoost)** para predição da tendência de desempenho acadêmico de estudantes do ensino médio da rede pública estadual do Piauí, a partir de um conjunto de dados sintético parametrizado em estatísticas oficiais (IBGE, INEP, PNAD Contínua).

## Propósito

O objetivo deste trabalho é construir um modelo de **prognóstico acadêmico** que classifique estudantes em:

- **Aprovado** - trajetória de aprovação ao final do ano letivo
- **Em Risco** - projeção na zona de fronteira dos critérios de aprovação
- **Reprovado** - projeção abaixo dos critérios da LDB

As variáveis acadêmicas observáveis são **parciais do 1º semestre**; a variável alvo (`situacao`) é a **tendência ao final do ano letivo**, derivada de uma projeção que combina o sinal parcial, a deriva do 2º semestre (fatores socioeconômicos e de engajamento) e ruído idiossincrático, rotulada pelos critérios da LDB (média 6,0 / frequência 75%).

## Etapas do estudo

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. Geração de dados sintéticos (generate_data.py)               │
│   - Features geradas primeiro (nota, frequência, renda...)      │
│   - situacao DERIVADA da projeção de fim de ano (LDB) + ruído   │
│   - Matriz de rastreabilidade parâmetro -> fonte oficial        │
├─────────────────────────────────────────────────────────────────┤
│ 2. Treinamento dos modelos (models.py)                          │
│   - Decision Tree + Random Forest + XGBoost                     │
│   - Baselines: Dummy majoritário + Regressão Logística          │
│   - SMOTE para balanceamento (dentro de cada fold)              │
│   - Validação cruzada estratificada (k=10) + GridSearchCV       │
│   - Teste de Wilcoxon pareado sobre os folds                    │
├─────────────────────────────────────────────────────────────────┤
│ 3. Execução dos notebooks (models/*.ipynb)                      │
│   - Análise exploratória + visualizações                        │
│   - Salvos com timestamp para histórico                         │
├─────────────────────────────────────────────────────────────────┤
│ 4. Relatório de métricas                                        │
│   - Tabelas comparativas (holdout + CV) e gráficos              │
│   - Salvo em estudo/output/result-*.md                          │
└─────────────────────────────────────────────────────────────────┘
```

## Estrutura

```text
.
├── README.md
├── referencias/               # Referências utilizadas
├── documento/                 # Documento LaTeX principal (artigo ABNT/IFPI)
│   ├── documento.tex          # Arquivo principal
│   ├── Makefile               # Build: make all (pdf + docx)
│   ├── output/                # PDF e DOCX gerados (gitignored)
│   ├── bibliografia.bib       # Referências bibliográficas
│   ├── abntex-ifpi/           # Customizações IFPI (estilo, capa, folha de aprovação)
│   ├── capitulos/             # Introdução, referencial, metodologia, resultados, conclusão
│   ├── configuracoes/         # Metadados, citações, cores, tipografia
│   ├── estrutura/             # Preâmbulo, pré-textual, textual, pós-textual
│   ├── pre-textual/           # Resumo, abstract, listas, siglas
│   ├── pos-textual/           # Referências, apêndices, anexos
│   └── imagens/               # Figuras do documento
├── estudo/                    # Experimentos (ML)
│   ├── generate_data.py       # Geração de dados sintéticos + rastreabilidade
│   ├── models.py              # Treino, avaliação, baselines, ablation, Wilcoxon
│   ├── noise_sweep.py         # Análise de sensibilidade ao ruído
│   ├── run.py                 # Orquestrador principal
│   ├── run.sh                 # Script de execução rápida
│   ├── requirements.txt       # Dependências Python
│   ├── models/                # Notebooks (cópias com timestamp)
│   ├── data/                  # Dataset + matriz de fontes + tabelas exportadas
│   └── output/                # Relatórios de métricas (gitignored)
└── .github/workflows/
    └── build-latex.yml        # CI: build PDF + DOCX
```

## Pré-requisitos

- Python 3.10+

## Instalação

```bash
cd estudo

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Registrar kernel do Jupyter (para execução dos notebooks)
python -m ipykernel install --user --name=tcc-estudo

# Configurar variáveis de ambiente
cp .env.example .env
```

## Como Rodar

### Execução completa (recomendado)

```bash
cd estudo
./run.sh
```

Ou manualmente:

```bash
source .venv/bin/activate
python run.py
```

Isso executa todas as etapas:

1. Carrega (ou gera) o dataset sintético em `data/`
2. Treina DT, RF e XGBoost com SMOTE + CV + GridSearchCV
3. Avalia baselines e roda o Wilcoxon pareado sobre os folds
4. Executa notebooks → `models/*_<timestamp>.ipynb`
5. Salva o relatório de métricas → `output/result-*.md`

## Documento LaTeX

O artigo/TCC em LaTeX fica inteiramente dentro de `documento/`.

- Arquivo principal: `documento/documento.tex`
- Build local: `cd documento && make all`
- Saída gerada: `documento/output/documento.pdf` e `documento/output/documento.docx`
- Mermaid no LaTeX: compilação via `lualatex -shell-escape`

### Apenas gerar dados

```bash
python generate_data.py
```

## Configuração (.env)

Todas as distribuições são configuráveis. Principais variáveis:

| Variável           | Descrição                                        | Default      |
| ------------------ | ------------------------------------------------ | ------------ |
| `N_SAMPLES`        | Quantidade de amostras                           | 3000         |
| `SEED`             | Semente aleatória                                | 42           |
| `LDB_MEDIA_MINIMA` | Média mínima de aprovação (LDB)                  | 6.0          |
| `LDB_FREQ_MINIMA`  | Frequência mínima de aprovação (LDB, %)          | 75           |
| `BANDA_RISCO_*`    | Banda pedagógica "Em Risco" (média e frequência) | 5.0-6.5 / 80 |
| `NOISE_LEVEL`      | Ruído idiossincrático da deriva do 2º semestre   | 3.0          |

Veja `.env.example` para a lista completa. Cenários de contraste: `.env.nacional`, `.env.publica`, `.env.elite` (uso: `./run.sh --env .env.nacional`).

## Como funciona a derivação do alvo

A variável `situacao` **não é gerada aleatoriamente** nem determinada pelo parcial do 1º semestre. A projeção de fim de ano combina:

| Componente              | Papel                                                                 |
| ----------------------- | --------------------------------------------------------------------- |
| Sinal acadêmico parcial | nota, frequência, média/taxa de entrega de atividades (1º sem.)       |
| Deriva do 2º semestre   | engajamento, renda, trabalho, conectividade, assistência estudantil   |
| Interações não lineares | efeito de retenção do Pé-de-Meia, dupla penalidade trabalho×distorção |
| Ruído idiossincrático   | fatores não observáveis (saúde, eventos familiares, motivação)        |

A rotulagem aplica os critérios da LDB (média 6,0 e frequência 75%) sobre a projeção, com banda pedagógica fixa para "Em Risco". Gênero e cor/raça têm peso zero no fenômeno gerador (variáveis apenas descritivas, excluídas do treino).

## Tecnologias

- **Python** - linguagem principal
- **scikit-learn** - modelos de ML e métricas
- **xgboost** - Gradient Boosting
- **imbalanced-learn** - SMOTE para balanceamento de classes
- **scipy** - teste de Wilcoxon
- **pandas / numpy** - manipulação de dados
- **matplotlib / seaborn** - visualizações
- **Jupyter** - notebooks interativos
