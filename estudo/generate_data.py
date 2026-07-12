"""
Script para geração de dados sintéticos de desempenho acadêmico.
Gera um CSV simulando uma turma média de ensino médio da REDE PÚBLICA
ESTADUAL DO PIAUÍ (cenário base; contrastes: nacional, alta
vulnerabilidade e rede privada - ver .env.* e data/auditoria_fontes.md).
Variável alvo: situacao (Aprovado, Em Risco, Reprovado) - TENDÊNCIA de
desempenho ao FINAL DO ANO LETIVO, rotulada pelos critérios da LDB.

SEMÂNTICA TEMPORAL: as variáveis acadêmicas observáveis (nota_media,
frequencia, media_atividades, taxa_entrega_atividades) são PARCIAIS DO
1º SEMESTRE. O desfecho anual é projetado a partir do sinal parcial +
deriva do 2º semestre (termos socioeconômicos e de engajamento) + ruído
idiossincrático - o parcial NÃO determina o final, preservando espaço
para intervenção pedagógica preventiva.

ROTULAGEM (LDB, Lei nº 9.394/1996, art. 24): aprovação exige média
final >= 6,0 E frequência >= 75%. "Em Risco" é uma banda pedagógica
fixa de fronteira (média projetada em [5,0; 6,5) ou frequência projetada
em [75; 80)) - decisão de modelagem, proporções de classe emergentes.

Abordagem: as features são geradas primeiro (com distribuições calibradas
em estatísticas oficiais - IBGE/PNAD, Censo Escolar/INEP, PNAD Contínua TIC
e literatura educacional) e `situacao` é DERIVADA da projeção de fim de
ano, com interações não lineares (ex.: efeito de retenção do Pé-de-Meia)
e outliers de resiliência acadêmica.

Diretrizes seguidas (ver referencias/other/Justificando Dados Sintéticos
Socioeconômicos.md):
- Sem determinismo linear: vulnerabilidade não implica reprovação inevitável;
- Assistência financeira (Pé-de-Meia) modelada como REGRA DE ELEGIBILIDADE
  (Lei nº 14.818), não como sorteio;
- Idade com assimetria positiva (distorção idade-série), não normal fixa;
- Gênero e cor/raça gerados para caracterização da turma, com PESO ZERO no
  score (proteção contra viés discriminatório);
- Rastreabilidade parâmetro -> fonte exportada junto com o CSV.

Todas as distribuições são configuráveis via arquivo .env
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# --- Configuração via .env ---
SEED = env_int("SEED", 42)
N_SAMPLES = env_int("N_SAMPLES", 500)

# Limiares de rotulagem ancorados na LDB (Lei nº 9.394/1996, art. 24):
# aprovação = média final >= 6,0 E frequência mínima de 75%
LDB_MEDIA_MINIMA = env_float("LDB_MEDIA_MINIMA", 6.0)
LDB_FREQ_MINIMA = env_float("LDB_FREQ_MINIMA", 75)
# Banda pedagógica fixa de fronteira ("Em Risco") - decisão de modelagem
BANDA_RISCO_MEDIA_INF = env_float("BANDA_RISCO_MEDIA_INF", 5.0)
BANDA_RISCO_MEDIA_SUP = env_float("BANDA_RISCO_MEDIA_SUP", 6.5)
BANDA_RISCO_FREQ_SUP = env_float("BANDA_RISCO_FREQ_SUP", 80)

# --- Renda familiar PER CAPITA (faixas em salários mínimos) ---
# ~60% dos estudantes de 15-17 anos da rede pública em situação de pobreza
PCT_RENDA_ATE_MEIO_SM = env_float("PORCENT_RENDA_PC_ATE_MEIO_SM", 60) / 100
PCT_RENDA_MEIO_A_1SM = env_float("PORCENT_RENDA_PC_MEIO_A_1SM", 25) / 100
PCT_RENDA_ACIMA_1SM = env_float("PORCENT_RENDA_PC_ACIMA_1SM", 15) / 100

# --- Demográficos (base PI - Censo 2022: pretos+pardos ~77%) ---
PCT_MASCULINO = env_float("PORCENT_MASCULINO", 49) / 100
PCT_PARDO = env_float("PORCENT_PARDO", 65) / 100
PCT_PRETO = env_float("PORCENT_PRETO", 12) / 100
PCT_BRANCO = env_float("PORCENT_BRANCO", 22) / 100
PCT_OUTRAS_COR = env_float("PORCENT_AMARELO_INDIGENA", 1) / 100
PCT_SOLTEIRO = env_float("PORCENT_SOLTEIRO", 93) / 100    # faixa etária 14-24
PCT_CASADO_UNIAO = env_float("PORCENT_CASADO_UNIAO", 5) / 100
PCT_EC_OUTRO = env_float("PORCENT_ESTADO_CIVIL_OUTRO", 2) / 100

# --- Distribuição por série (funil de matrículas do EM) ---
PCT_SERIE_1 = env_float("PORCENT_SERIE_1", 38) / 100
PCT_SERIE_2 = env_float("PORCENT_SERIE_2", 33) / 100
PCT_SERIE_3 = env_float("PORCENT_SERIE_3", 29) / 100

# --- Distorção idade-série (2+ anos de atraso) ---
PCT_DISTORCAO = env_float("PORCENT_DISTORCAO_IDADE_SERIE", 21.7) / 100

# --- Socioeconômicos ---
PCT_ESCOLA_PUB = env_float("PORCENT_ESCOLA_PUBLICA", 85) / 100
PCT_TRABALHA = env_float("PORCENT_TRABALHA", 26.3) / 100
PCT_INTERNET = env_float("PORCENT_ACESSO_INTERNET", 92.5) / 100
PCT_COMPUTADOR = env_float("PORCENT_POSSUI_COMPUTADOR", 25.5) / 100

# --- Assistência financeira (Pé-de-Meia, Lei nº 14.818) ---
TAXA_ADESAO_PE_DE_MEIA = env_float("TAXA_ADESAO_PE_DE_MEIA", 85) / 100
PE_DE_MEIA_FREQ_MINIMA = env_float("PE_DE_MEIA_FREQ_MINIMA", 80)

# --- Outliers de resiliência acadêmica ---
PCT_OUTLIERS_RESILIENCIA = env_float("PORCENT_OUTLIERS_RESILIENCIA", 3) / 100

# --- Acadêmicos ---
NOTA_MEDIA_BASE = env_float("NOTA_MEDIA_BASE", 6.2)
NOTA_DESVIO = env_float("NOTA_DESVIO", 1.3)

# Ruído idiossincrático da deriva do 2º semestre (variância não observável:
# saúde, eventos familiares, motivação). Escala: décimos de ponto de média
# (NOISE_LEVEL=3.0 -> sigma=0,30 na média final projetada; a frequência
# projetada usa sigma = NOISE_LEVEL * 0,8 p.p.)
NOISE_LEVEL = env_float("NOISE_LEVEL", 3.0)

np.random.seed(SEED)

RENDA_LABELS = ["Ate 0,5 SM", "0,5-1 SM", "Acima de 1 SM"]

# Multiplicadores relativos por faixa de renda (mantêm o gradiente observado
# na PNAD Contínua TIC, com a média geral ancorada no alvo do .env)
_INTERNET_FATORES = np.array([0.93, 1.075, 1.135])
_COMPUTADOR_FATORES = np.array([0.60, 1.35, 2.25])


def _normalize_probs(*probs) -> list[float]:
    """Normaliza probabilidades para somar exatamente 1.0."""
    total = sum(probs)
    return [p / total for p in probs]


def _prob_condicional_por_renda(renda_idx: np.ndarray, alvo: float, fatores: np.ndarray) -> np.ndarray:
    """Deriva probabilidades condicionais por faixa de renda de modo que a
    média ponderada realize (aprox.) o alvo global configurado no .env."""
    shares = np.array([(renda_idx == i).mean() for i in range(3)])
    denom = float((shares * fatores).sum())
    base = alvo / denom if denom > 0 else alvo
    p_por_faixa = np.clip(base * fatores, 0.01, 0.99)
    return p_por_faixa[renda_idx]


def generate_synthetic_data(n: int = N_SAMPLES) -> pd.DataFrame:
    # === 1. GERAR FEATURES ===

    # --- Série e idade (assimetria positiva via distorção idade-série) ---
    serie = np.random.choice(
        [1, 2, 3], size=n, p=_normalize_probs(PCT_SERIE_1, PCT_SERIE_2, PCT_SERIE_3)
    )
    # Idade regular: 15 anos na 1ª série, 16 na 2ª, 17 na 3ª (+0/1 de variação)
    idade_regular = 14 + serie + np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    distorcao = np.random.random(n) < PCT_DISTORCAO
    # Cauda longa de atraso: 2 a 7 anos, decaindo - limite superior de 24 anos
    atraso = np.random.choice([2, 3, 4, 5, 6, 7], size=n, p=[0.42, 0.25, 0.15, 0.09, 0.06, 0.03])
    idade = np.where(distorcao, idade_regular + atraso, idade_regular).clip(14, 24).astype(int)

    # --- Demográficos (caracterização; PESO ZERO no score) ---
    genero = np.random.choice(
        ["Masculino", "Feminino"], size=n, p=[PCT_MASCULINO, 1 - PCT_MASCULINO]
    )
    cor_raca = np.random.choice(
        ["Parda", "Preta", "Branca", "Outra"], size=n,
        p=_normalize_probs(PCT_PARDO, PCT_PRETO, PCT_BRANCO, PCT_OUTRAS_COR),
    )
    estado_civil = np.random.choice(
        ["Solteiro", "Casado/Uniao", "Outro"], size=n,
        p=_normalize_probs(PCT_SOLTEIRO, PCT_CASADO_UNIAO, PCT_EC_OUTRO),
    )

    # --- Socioeconômicos ---
    renda_idx = np.random.choice(
        3, size=n,
        p=_normalize_probs(PCT_RENDA_ATE_MEIO_SM, PCT_RENDA_MEIO_A_1SM, PCT_RENDA_ACIMA_1SM),
    )
    renda_per_capita = np.array(RENDA_LABELS)[renda_idx]

    tipo_escola_origem = np.random.choice(
        ["Publica", "Privada"], size=n, p=[PCT_ESCOLA_PUB, 1 - PCT_ESCOLA_PUB]
    )

    # Conectividade condicionada à renda (média global = alvo do .env)
    p_internet = _prob_condicional_por_renda(renda_idx, PCT_INTERNET, _INTERNET_FATORES)
    acesso_internet = np.where(np.random.random(n) < p_internet, "Sim", "Nao")

    p_computador = _prob_condicional_por_renda(renda_idx, PCT_COMPUTADOR, _COMPUTADOR_FATORES)
    possui_computador = np.where(np.random.random(n) < p_computador, "Sim", "Nao")

    # Trabalho: mais provável para maiores de idade / em distorção
    share_18 = (idade >= 18).mean()
    fator_trabalha = np.where(idade >= 18, 2.2, 0.7)
    base_trabalha = PCT_TRABALHA / (share_18 * 2.2 + (1 - share_18) * 0.7)
    trabalha = np.where(
        np.random.random(n) < np.clip(base_trabalha * fator_trabalha, 0.01, 0.95), "Sim", "Nao"
    )

    # === 2. VARIÁVEIS ACADÊMICAS - PARCIAIS DO 1º SEMESTRE ===
    # (influenciadas moderadamente pelo contexto socioeconômico)
    nota_media = np.random.normal(NOTA_MEDIA_BASE, NOTA_DESVIO, size=n)
    nota_media += np.where(possui_computador == "Sim", 0.35, 0.0)
    nota_media += np.where(acesso_internet == "Sim", 0.25, 0.0)
    nota_media += np.where(trabalha == "Sim", -0.45, 0.0)
    nota_media += np.where(distorcao, -0.55, 0.0)
    nota_media += np.where(tipo_escola_origem == "Privada", 0.15, 0.0)
    nota_media += np.where(renda_idx == 2, 0.20, 0.0)
    nota_media = nota_media.clip(0, 10)

    # Frequência base correlacionada com a nota
    frequencia = 52 + nota_media * 4.8 + np.random.normal(0, 6.5, size=n)

    # === 3. PÉ-DE-MEIA COMO REGRA DE ELEGIBILIDADE (Lei nº 14.818) ===
    # Elegível: 14-24 anos + renda per capita até 0,5 SM (proxy de CadÚnico)
    elegivel_pdm = (renda_idx == 0) & (idade >= 14) & (idade <= 24)
    aderiu_pdm = elegivel_pdm & (np.random.random(n) < TAXA_ADESAO_PE_DE_MEIA)

    # Efeito NÃO LINEAR de retenção: o incentivo condicionado à presença
    # (mín. 80%/mês) puxa a frequência dos aderentes para cima
    lift = np.maximum(0, (PE_DE_MEIA_FREQ_MINIMA + 2) - frequencia) * 0.75 + 2.0
    frequencia = np.where(aderiu_pdm, frequencia + lift, frequencia)
    frequencia = frequencia.clip(30, 100).round(1)

    # Beneficiário efetivo: aderiu E mantém a frequência mínima
    beneficiario_pdm = aderiu_pdm & (frequencia >= PE_DE_MEIA_FREQ_MINIMA)
    assistencia_pe_de_meia = np.where(beneficiario_pdm, "Sim", "Nao")

    # Reprovações acumuladas no EM (0 a 3): inversas à nota; distorção implica >= 1
    prob_reprov = ((10 - nota_media) / 14).clip(0.02, 0.6)
    reprovacoes = np.array([np.random.binomial(n=2, p=p) for p in prob_reprov])
    reprovacoes = np.where(distorcao, np.maximum(reprovacoes, 1), reprovacoes)
    reprovacoes = reprovacoes.clip(0, 3)

    # === 3b. ENGAJAMENTO CONTÍNUO (novas variáveis observáveis, 1º sem.) ===
    # Fator latente de engajamento: parcialmente independente da nota -
    # capta o comportamento contínuo (entregas), não só o desempenho em provas.
    # Decisão de modelagem ancorada na literatura de engajamento escolar
    # (ver matriz de rastreabilidade).
    engajamento = np.random.normal(0, 1, size=n)
    engajamento += np.where(beneficiario_pdm, 0.45, 0.0)   # condicionalidade de presença
    engajamento += np.where(trabalha == "Sim", -0.35, 0.0)  # menos tempo disponível
    engajamento += np.where(distorcao, -0.25, 0.0)

    # taxa_entrega_atividades (%): proporção de atividades entregues no 1º sem.
    # Sinal de engajamento INDEPENDENTE da nota (dá para entregar tudo com nota
    # baixa, ou pouco com nota alta).
    taxa_entrega_atividades = (
        62 + engajamento * 11 + (frequencia - 85) * 0.55
        + np.random.normal(0, 7, size=n)
    ).clip(10, 100).round(1)

    # media_atividades (0-10): média das notas de atividades NORMALIZADAS para
    # a escala 0-10 (ex.: atividade que vale 2,0 com nota 1,0 -> 5,0).
    # Correlacionada com a nota de provas, com ruído próprio de engajamento.
    media_atividades = (
        nota_media * 0.65 + 1.6 + engajamento * 0.75
        + np.random.normal(0, 0.7, size=n)
    ).clip(0, 10).round(1)

    # === 4. OUTLIERS DE RESILIÊNCIA ACADÊMICA ===
    # Perfis de alta vulnerabilidade com desempenho alto (ex.: Cocal dos Alves-PI):
    # impedem o determinismo social nas fronteiras de decisão do modelo
    vulneraveis = np.where((renda_idx == 0) & (possui_computador == "Nao"))[0]
    k = min(int(round(n * PCT_OUTLIERS_RESILIENCIA)), len(vulneraveis))
    if k > 0:
        idx_resil = np.random.choice(vulneraveis, size=k, replace=False)
        nota_media[idx_resil] = np.random.uniform(8.5, 9.7, size=k)
        frequencia[idx_resil] = np.random.uniform(90, 100, size=k).round(1)
        reprovacoes[idx_resil] = np.where(distorcao[idx_resil], 1, 0)
        media_atividades[idx_resil] = np.random.uniform(8.3, 10.0, size=k).round(1)
        taxa_entrega_atividades[idx_resil] = np.random.uniform(88, 100, size=k).round(1)
        engajamento[idx_resil] = np.abs(engajamento[idx_resil]) + 0.5

    nota_media = nota_media.round(1)

    # === 5. DERIVAR VARIÁVEL ALVO (situacao) - PROJEÇÃO DE FIM DE ANO ===
    # O desfecho anual = sinal parcial do 1º sem. + DERIVA do 2º semestre
    # (termos socioeconômicos e de engajamento) + ruído idiossincrático.
    # A deriva é mapeada para a ESCALA DE MÉDIA (pontos de 0-10) e para a
    # escala de frequência (p.p.), dando significado pedagógico direto aos
    # limiares da LDB.

    # Deriva da MÉDIA no 2º semestre (pontos de média)
    deriva_nota = np.zeros(n, dtype=float)
    deriva_nota += engajamento * 0.28                       # entrega contínua sustenta a média
    deriva_nota += (media_atividades - nota_media) * 0.10   # atividades acima das provas puxam para cima
    deriva_nota += np.array([-0.15, 0.05, 0.15])[renda_idx]
    deriva_nota += np.where(possui_computador == "Sim", 0.12, 0.0)
    deriva_nota += np.where(acesso_internet == "Sim", 0.10, -0.05)
    deriva_nota += np.where(trabalha == "Sim", -0.18, 0.0)
    deriva_nota += np.where(distorcao, -0.12, 0.0)
    # Interações NÃO LINEARES (anti-determinismo)
    deriva_nota += np.where(beneficiario_pdm, 0.30, 0.0)    # retenção do Pé-de-Meia
    deriva_nota -= np.where((trabalha == "Sim") & distorcao, 0.15, 0.0)
    # Gênero e cor/raça: PESO ZERO (variáveis apenas descritivas)
    # Ruído idiossincrático (saúde, eventos familiares, motivação)
    deriva_nota += np.random.normal(0, NOISE_LEVEL / 10.0, size=n)

    media_final_projetada = (nota_media + deriva_nota).clip(0, 10)

    # Deriva da FREQUÊNCIA no 2º semestre (pontos percentuais)
    deriva_freq = np.zeros(n, dtype=float)
    deriva_freq += engajamento * 1.6
    deriva_freq += np.where(beneficiario_pdm, 2.5, 0.0)     # condicionalidade de 80%/mês
    deriva_freq += np.where(trabalha == "Sim", -1.8, 0.0)
    deriva_freq += np.where(distorcao, -1.2, 0.0)
    deriva_freq += np.random.normal(0, NOISE_LEVEL * 0.8, size=n)

    frequencia_final_projetada = (frequencia + deriva_freq).clip(20, 100)

    # Rotulagem pelos critérios da LDB (art. 24) + banda pedagógica fixa:
    # - Reprovado: média final < banda inferior (5,0) OU frequência < 75%
    # - Em Risco: fronteira - média em [5,0; 6,5) OU frequência em [75; 80)
    # - Aprovado: média >= 6,5 E frequência >= 80 (fora da zona de fronteira)
    reprovado = (
        (media_final_projetada < BANDA_RISCO_MEDIA_INF)
        | (frequencia_final_projetada < LDB_FREQ_MINIMA)
    )
    em_risco = ~reprovado & (
        (media_final_projetada < BANDA_RISCO_MEDIA_SUP)
        | (frequencia_final_projetada < BANDA_RISCO_FREQ_SUP)
    )
    situacao = np.where(reprovado, "Reprovado", np.where(em_risco, "Em Risco", "Aprovado"))

    df = pd.DataFrame(
        {
            "serie": serie,
            "idade": idade,
            "genero": genero,
            "cor_raca": cor_raca,
            "estado_civil": estado_civil,
            "renda_per_capita": renda_per_capita,
            "trabalha": trabalha,
            "acesso_internet": acesso_internet,
            "possui_computador": possui_computador,
            "tipo_escola_origem": tipo_escola_origem,
            "distorcao_idade_serie": np.where(distorcao, "Sim", "Nao"),
            "assistencia_pe_de_meia": assistencia_pe_de_meia,
            # Parciais do 1º semestre (observáveis no momento da predição)
            "nota_media": nota_media,
            "frequencia": frequencia,
            "media_atividades": media_atividades,
            "taxa_entrega_atividades": taxa_entrega_atividades,
            "reprovacoes": reprovacoes,
            # Alvo: tendência ao final do ano letivo (critérios LDB)
            "situacao": situacao,
        }
    )

    return df


# ============================================================
# Matriz de rastreabilidade:
# parâmetro -> valor -> evidência -> fonte -> nível de desagregação
# Níveis: "estadual (PI)" | "regional (NE)" | "nacional" |
#         "decisão de modelagem" - proxies levam nota explícita
# (auditoria completa em data/auditoria_fontes.md)
# ============================================================
def build_traceability() -> list[dict]:
    return [
        {
            "parametro": "PORCENT_ESCOLA_PUBLICA",
            "valor": PCT_ESCOLA_PUB * 100,
            "evidencia": "PI: 715.748 estudantes na rede pública (2025); no BR, 82,0% das matrículas do EM são da rede pública",
            "fonte": "INEP - Sinopse Estatística do Censo Escolar / Todos Pela Educação (Panorama Piauí)",
            "nivel": "estadual (PI)",
        },
        {
            "parametro": "PORCENT_RENDA_PC_ATE_MEIO_SM",
            "valor": PCT_RENDA_ATE_MEIO_SM * 100,
            "evidencia": "PROXY: SIS/IBGE - 45,3% da população geral do PI em pobreza (2023, linha ~0,5 SM); pobreza <18 anos é maior que a geral (BR: 45,6%; NE: ~2/3 em 2022)",
            "fonte": "IBGE - Síntese de Indicadores Sociais 2024 / Fundação Abrinq - Um Retrato da Infância e Adolescência no Brasil (PNAD Contínua)",
            "nivel": "estadual (PI) - proxy: população geral, não estudantes 15-17",
        },
        {
            "parametro": "PORCENT_MASCULINO / PORCENT_PARDO+PRETO",
            "valor": f"{PCT_MASCULINO*100:.0f} / {(PCT_PARDO+PCT_PRETO)*100:.0f}",
            "evidencia": "PI (Censo 2022): pardos 64,8%, pretos 12,3% (77,1% pretos+pardos), brancos 22,6%; razão de sexo 95,81 homens/100 mulheres",
            "fonte": "IBGE - Censo Demográfico 2022 (divulgação cor ou raça)",
            "nivel": "estadual (PI)",
        },
        {
            "parametro": "PORCENT_TRABALHA",
            "valor": PCT_TRABALHA * 100,
            "evidencia": "PROXY: 26,3% dos alunos do EM de baixa renda conciliam trabalho e estudo (literatura); PI 5-17 anos: 6,8-8,6% em trabalho infantil (conceito distinto); BR 16-17: 15,3% (2024)",
            "fonte": "SciELO/CEBAPE / MTE - Diagnóstico Ligeiro do Trabalho Infantil na PNADc 2023-2024",
            "nivel": "decisão de modelagem - ancorada em literatura nacional + taxa PI de conceito adjacente",
        },
        {
            "parametro": "PORCENT_ACESSO_INTERNET",
            "valor": PCT_INTERNET * 100,
            "evidencia": "PI: 92,5% dos domicílios com utilização de internet (2025; 88,9% em 2023)",
            "fonte": "IBGE - PNAD Contínua TIC, SIDRA tabela 7307",
            "nivel": "estadual (PI)",
        },
        {
            "parametro": "PORCENT_POSSUI_COMPUTADOR",
            "valor": PCT_COMPUTADOR * 100,
            "evidencia": "PI: 25,5% dos domicílios com microcomputador ou tablet (2025); BR: 40,9% - exclusão de dispositivo acentuada no PI",
            "fonte": "IBGE - PNAD Contínua TIC, SIDRA tabela 7302",
            "nivel": "estadual (PI)",
        },
        {
            "parametro": "PORCENT_DISTORCAO_IDADE_SERIE",
            "valor": PCT_DISTORCAO * 100,
            "evidencia": "Distorção idade-série de 21,7% na rede pública estadual do PI (histórico 2022: 33,2%); idade com assimetria positiva até 24 anos",
            "fonte": "INEP - Taxas de Distorção Idade-série (dados abertos) / Todos Pela Educação (PDF Piauí)",
            "nivel": "estadual (PI)",
        },
        {
            "parametro": "TAXA_ADESAO_PE_DE_MEIA / PE_DE_MEIA_FREQ_MINIMA",
            "valor": f"{TAXA_ADESAO_PE_DE_MEIA*100:.0f} / {PE_DE_MEIA_FREQ_MINIMA:.0f}",
            "evidencia": "Elegibilidade: 14-24 anos, CadÚnico com renda per capita <= 0,5 SM, frequência mínima de 80%/mês; PI: 187.620 atendidos, abandono 8,6% -> 0,5%",
            "fonte": "Lei nº 14.818 (Pé-de-Meia) - CAIXA / Portal da Transparência / MDS Informe CadÚnico nº 50 / Cadernos Cajuína",
            "nivel": "estadual (PI) - regra nacional com efeito documentado no PI",
        },
        {
            "parametro": "PORCENT_OUTLIERS_RESILIENCIA",
            "valor": PCT_OUTLIERS_RESILIENCIA * 100,
            "evidencia": "Perfis vulneráveis de alto desempenho (ex.: U.E. Augustinho Brandão, Cocal dos Alves-PI) validam fronteiras de decisão sem determinismo social",
            "fonte": "SEDUC-PI / Fundação Lemann",
            "nivel": "decisão de modelagem - âncora qualitativa estadual (PI)",
        },
        {
            "parametro": "LDB_MEDIA_MINIMA / LDB_FREQ_MINIMA",
            "valor": f"{LDB_MEDIA_MINIMA:.1f} / {LDB_FREQ_MINIMA:.0f}",
            "evidencia": "Critérios reais de aprovação no ensino médio: média final >= 6,0 e frequência mínima de 75% da carga horária",
            "fonte": "LDB - Lei nº 9.394/1996, art. 24 (frequência); média 6,0 como padrão das redes estaduais",
            "nivel": "normativo (nacional)",
        },
        {
            "parametro": "BANDA_RISCO_MEDIA (5,0-6,5) / BANDA_RISCO_FREQ (75-80)",
            "valor": f"{BANDA_RISCO_MEDIA_INF:.1f}-{BANDA_RISCO_MEDIA_SUP:.1f} / {LDB_FREQ_MINIMA:.0f}-{BANDA_RISCO_FREQ_SUP:.0f}",
            "evidencia": "Banda pedagógica fixa de fronteira para a classe 'Em Risco' (projeção anual próxima dos limiares da LDB); proporções de classe emergentes, não calibradas",
            "fonte": "Definição metodológica do estudo",
            "nivel": "decisão de modelagem",
        },
        {
            "parametro": "MEDIA_ATIVIDADES / TAXA_ENTREGA_ATIVIDADES",
            "valor": "derivadas (0-10 normalizada / %)",
            "evidencia": "Sinais de engajamento contínuo do 1º semestre: notas de atividades normalizadas para 0-10 e proporção de atividades entregues; correlacionadas a nota/frequência com ruído próprio",
            "fonte": "Decisão de modelagem ancorada na literatura de engajamento escolar (EDM)",
            "nivel": "decisão de modelagem",
        },
        {
            "parametro": "NOISE_LEVEL",
            "valor": f"{NOISE_LEVEL:.1f}",
            "evidencia": "Variância idiossincrática da deriva do 2º semestre (fatores não observáveis: saúde, eventos familiares, motivação); sigma = NOISE_LEVEL/10 pontos de média e NOISE_LEVEL*0,8 p.p. de frequência",
            "fonte": "Definição metodológica do estudo",
            "nivel": "decisão de modelagem",
        },
    ]


def save_traceability(output_dir: str = "data") -> tuple[str, str]:
    """Exporta a matriz de rastreabilidade parâmetro -> fonte em JSON e MD."""
    import json

    os.makedirs(output_dir, exist_ok=True)
    rows = build_traceability()

    json_path = os.path.join(output_dir, "parametros_fontes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(output_dir, "parametros_fontes.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Matriz de rastreabilidade - parâmetros do gerador sintético\n\n")
        f.write("Cenário base: rede pública estadual do Piauí "
                "(auditoria de fontes em `auditoria_fontes.md`).\n\n")
        f.write("| Parâmetro | Valor usado | Evidência | Fonte | Nível de desagregação |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                f"| `{r['parametro']}` | {r['valor']} | {r['evidencia']} "
                f"| {r['fonte']} | {r['nivel']} |\n"
            )

    return json_path, md_path


def print_distribution_check(df: pd.DataFrame) -> None:
    """Imprime verificação das distribuições geradas (alvo vs. realizado)."""
    n = len(df)
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DAS DISTRIBUIÇÕES")
    print("=" * 60)

    # Situação (derivada da projeção de fim de ano; proporções emergentes)
    print("\n--- Variável alvo (situacao) - DERIVADA (rotulagem LDB) ---")
    for classe in ["Aprovado", "Em Risco", "Reprovado"]:
        count = (df["situacao"] == classe).sum()
        pct = count / n * 100
        print(f"  {classe:12s}: {count:4d} ({pct:5.1f}%)")
    print(f"  LDB: média >= {LDB_MEDIA_MINIMA}, frequência >= {LDB_FREQ_MINIMA}% | "
          f"banda 'Em Risco': média [{BANDA_RISCO_MEDIA_INF}; {BANDA_RISCO_MEDIA_SUP}) "
          f"ou freq [{LDB_FREQ_MINIMA}; {BANDA_RISCO_FREQ_SUP})")

    # Renda per capita
    print("\n--- Renda per capita ---")
    for faixa, pct in zip(RENDA_LABELS, [PCT_RENDA_ATE_MEIO_SM, PCT_RENDA_MEIO_A_1SM, PCT_RENDA_ACIMA_1SM]):
        real = (df["renda_per_capita"] == faixa).sum() / n * 100
        print(f"  {faixa:15s}: {real:5.1f}% (alvo: {pct*100:.0f}%)")

    # Conectividade e trabalho
    print("\n--- Conectividade e trabalho (alvo global) ---")
    for col, alvo in [
        ("acesso_internet", PCT_INTERNET),
        ("possui_computador", PCT_COMPUTADOR),
        ("trabalha", PCT_TRABALHA),
    ]:
        real = (df[col] == "Sim").mean() * 100
        print(f"  {col:20s}: {real:5.1f}% (alvo: {alvo*100:.1f}%)")

    # Distorção idade-série e idade
    print("\n--- Idade e distorção idade-série ---")
    real_dist = (df["distorcao_idade_serie"] == "Sim").mean() * 100
    print(f"  distorcao_idade_serie: {real_dist:5.1f}% (alvo: {PCT_DISTORCAO*100:.1f}%)")
    print(f"  idade: média={df['idade'].mean():.1f}, mediana={df['idade'].median():.0f}, "
          f"máx={df['idade'].max()}, assimetria={df['idade'].skew():.2f}")

    # Pé-de-Meia
    print("\n--- Pé-de-Meia (regra de elegibilidade) ---")
    beneficiarios = (df["assistencia_pe_de_meia"] == "Sim").mean() * 100
    vulneraveis = (df["renda_per_capita"] == RENDA_LABELS[0]).mean() * 100
    print(f"  Vulneráveis (renda <= 0,5 SM): {vulneraveis:5.1f}%")
    print(f"  Beneficiários efetivos:        {beneficiarios:5.1f}%")

    # Acadêmicas (parciais do 1º semestre)
    print("\n--- Variáveis acadêmicas (parciais do 1º semestre) ---")
    print(f"  Nota média:  média={df['nota_media'].mean():.2f}, std={df['nota_media'].std():.2f}")
    print(f"  Frequência:  média={df['frequencia'].mean():.1f}%, std={df['frequencia'].std():.1f}")
    print(f"  Reprovações: média={df['reprovacoes'].mean():.2f}, std={df['reprovacoes'].std():.2f}")
    if "media_atividades" in df.columns:
        print(f"  Média ativid.: média={df['media_atividades'].mean():.2f}, std={df['media_atividades'].std():.2f} "
              f"(corr c/ nota: {df['media_atividades'].corr(df['nota_media']):.2f})")
        print(f"  Taxa entrega:  média={df['taxa_entrega_atividades'].mean():.1f}%, std={df['taxa_entrega_atividades'].std():.1f} "
              f"(corr c/ nota: {df['taxa_entrega_atividades'].corr(df['nota_media']):.2f})")


def generate_and_save(n: int = N_SAMPLES) -> tuple[pd.DataFrame, str]:
    """Gera dados sintéticos e salva com nome baseado em timestamp.
    Também cria cópia em dados_academicos.csv (nome fixo para notebooks)
    e exporta a matriz de rastreabilidade parâmetro -> fonte.
    Retorna (DataFrame, caminho_do_arquivo)."""
    from datetime import datetime

    df = generate_synthetic_data(n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dados_academicos_{timestamp}.csv"
    output_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Cópia com nome fixo para uso nos notebooks
    fixed_path = os.path.join("data", "dados_academicos.csv")
    df.to_csv(fixed_path, index=False, encoding="utf-8-sig")

    # Matriz de rastreabilidade parâmetro -> fonte
    json_path, md_path = save_traceability()
    print(f"Rastreabilidade exportada: {json_path}, {md_path}")

    return df, output_path


if __name__ == "__main__":
    df, output_path = generate_and_save(N_SAMPLES)

    print(f"Dataset gerado com {len(df)} amostras em '{output_path}'")
    print(f"\nDistribuição da variável alvo:")
    print(df["situacao"].value_counts())
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nEstatísticas:")
    print(df.describe())

    print_distribution_check(df)
