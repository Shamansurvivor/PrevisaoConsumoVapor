# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Random Forest — Vapor como saída, demais variáveis como entrada
# MAGIC
# MAGIC **Definição do sistema**
# MAGIC - **Saídas (alvos):** vapor gerado — `11FI208-02`, `11FI502-30`, `21FI550-30` (caldeiras a gás) e `21FI551-10` (biomassa),
# MAGIC   além da **demanda total** = soma das quatro.
# MAGIC - **Entradas:** todas as demais tags do `1_bronze` (gás, ar, pressões, temperaturas, níveis, analisadores…), clima e calendário.
# MAGIC   Totalizadores (`_TOT`) ficam de fora por serem somas acumuladas.
# MAGIC
# MAGIC **Dois modos de uso do mesmo modelo**
# MAGIC
# MAGIC | Modo | Pergunta | Entradas | Alvo |
# MAGIC |---|---|---|---|
# MAGIC | **A — Processo** (soft sensor) | Dado o que entrou na caldeira no dia, quanto vapor saiu? | entradas do **mesmo dia** | vapor de cada caldeira a gás no dia *t* |
# MAGIC | **B — Previsão** | Quanto vapor será demandado amanhã / na semana / no mês? | somente informação **até o dia *t*** | demanda total em *t+1*, soma *t+1…t+7*, soma *t+1…t+30* |
# MAGIC
# MAGIC O modo A valida a física (gás/ar → vapor) e alimenta a estimativa de gás. O modo B é a previsão propriamente dita.
# MAGIC
# MAGIC Validação temporal (sem embaralhar), com *gap* igual ao horizonte, e comparação com **linhas de base** ingênuas.

# COMMAND ----------

# DBTITLE 1,Imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# COMMAND ----------

# DBTITLE 1,Configuração
BRONZE_DIR = "/Workspace/Previsao de consumo de vapor/1_bronze/processo_dados"
CLIMA_PATH = ("/Workspace/Previsao de consumo de vapor/1_bronze/clima_dados/"
              "dados_meteorologicos_sao_jose_dos_campos_20212026.csv - "
              "dados_meteorologicos_sao_jose_dos_campos_20212026.csv")
RESULT_DIR = "/Workspace/Previsao de consumo de vapor/3_gold"

# SAÍDAS do sistema (vapor gerado)
SAIDAS = {'C208': '11FI208-02', 'C502': '11FI502-30', 'C550': '21FI550-30', 'BIO': '21FI551-10'}

# Entradas de combustão (só para agrupar a importância por categoria)
GAS = {'C208': '11FC208-106', 'C502': '11FC502-01', 'C550': '21FC550-72'}
AR  = {'C208': '11FC208-138', 'C502': '11FC502-04', 'C550': '21FC550-69'}

MIN_HORAS_DIA = 20                 # dia válido com >= 20 h de dado
MAX_FRAC_NAN_ENTRADA = 0.5         # entrada descartada se > 50% faltante no treino
FRAC_BIOMASSA_ON = 0.05            # biomassa "operando" acima de 5% do seu p95 diário

HORIZONTES = {'D+1': 1, 'Semana (7d)': 7, 'Mês (30d)': 30}   # modo B
ALVOS_MODO_A = ['C208', 'C502', 'C550']                      # modo A

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1)
N_SPLITS = 5            # dobras da validação em janela deslizante
FRAC_TESTE = 0.2        # teste final (último período) para importância

# COMMAND ----------

# DBTITLE 1,Carregar todas as tags do bronze (horário) e agregar por dia
SKIP_PATTERNS = [r'\d{4}-\d{2}-\d{2}']
norm = lambda t: t.upper().replace('.PV', '').strip()

arquivos = []
for root, _, files in os.walk(BRONZE_DIR):
    for f in files:
        if f.endswith('.csv') and not any(re.search(p, f) for p in SKIP_PATTERNS):
            arquivos.append(os.path.join(root, f))
arquivos.sort()

por_tag = {}
for f in arquivos:
    por_tag.setdefault(os.path.basename(f).rsplit('.csv', 1)[0], []).append(f)


def carregar(tag):
    frames = []
    for f in por_tag[tag]:
        df = pd.read_csv(f)
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        frames.append(df[['ts', 'value']].dropna(subset=['ts']))
    s = pd.concat(frames).drop_duplicates('ts').set_index('ts')['value'].sort_index()
    return s.resample('1h').mean()


tags = [t for t in por_tag if '_TOT' not in t.upper()]
h = pd.DataFrame({norm(t): carregar(t) for t in tags}).asfreq('h')
print(f"Tags carregadas (sem _TOT): {len(tags)} | horário: {h.shape} | {h.index.min()} a {h.index.max()}")

# Conferir que as saídas existem
faltam = [f"{k}={v}" for k, v in SAIDAS.items() if norm(v) not in h]
if faltam:
    print(">>> ATENÇÃO — saídas não encontradas no bronze:", faltam)

# Agregação diária: média do dia, válida só com >= MIN_HORAS_DIA horas
g = h.resample('D')
d_media = g.mean().where(g.count() >= MIN_HORAS_DIA)

# Clima (média diária)
try:
    cl = pd.read_csv(CLIMA_PATH)
    cl['datetime_local'] = pd.to_datetime(cl['datetime_local'])
    cl = cl.set_index('datetime_local').select_dtypes('number')
    cl = cl[~cl.index.duplicated()].resample('D').mean().add_prefix('clima_')
    d_media = d_media.join(cl, how='left')
    print(f"Clima: {list(cl.columns)}")
except Exception as e:
    print(f"Clima não carregado: {e}")

# COMMAND ----------

# DBTITLE 1,Saídas diárias (vapor/dia) e demanda total
Y = pd.DataFrame(index=d_media.index)
for k, tag in SAIDAS.items():
    if norm(tag) in d_media:
        Y[k] = d_media[norm(tag)] * 24          # total do dia

# Partida da biomassa e demanda total (biomassa = 0 antes da partida)
INICIO_BIOMASSA = None
if 'BIO' in Y and Y['BIO'].notna().any():
    lim = FRAC_BIOMASSA_ON * Y['BIO'].quantile(0.95)
    INICIO_BIOMASSA = Y.index[(Y['BIO'] > lim).fillna(False)].min()
    Y.loc[Y.index < INICIO_BIOMASSA, 'BIO'] = 0.0
print(f"Início da biomassa detectado: {INICIO_BIOMASSA}")
# INICIO_BIOMASSA = pd.Timestamp('2026-05-01')   # descomente para fixar manualmente

gas_boilers = [c for c in ['C208', 'C502', 'C550'] if c in Y]
Y['demanda_total'] = Y[gas_boilers + (['BIO'] if 'BIO' in Y else [])].sum(axis=1, min_count=len(gas_boilers) + ('BIO' in Y))

# Entradas = tudo que não é saída
saidas_norm = {norm(v) for v in SAIDAS.values()}
ENTRADAS = [c for c in d_media.columns if c not in saidas_norm]
X_base = d_media[ENTRADAS].copy()
X_base['biomassa_operando'] = (Y['BIO'] > 0).astype(float) if 'BIO' in Y else 0.0

print(f"Dias: {len(Y)} | demanda total válida em {Y['demanda_total'].notna().sum()} dias")
print(f"Entradas candidatas: {len(ENTRADAS)} tags/clima")

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(Y.index, Y['demanda_total'], lw=0.8, label='demanda total')
for c in gas_boilers + ['BIO']:
    if c in Y:
        ax.plot(Y.index, Y[c], lw=0.6, alpha=0.7, label=c)
if INICIO_BIOMASSA is not None:
    ax.axvline(INICIO_BIOMASSA, color='k', ls='--')
ax.legend(ncol=5)
ax.set_title('Saídas diárias de vapor')
plt.show()

# COMMAND ----------

# DBTITLE 1,Funções: categorias, métricas, modelo e avaliação
gas_n = {norm(v) for v in GAS.values()}
ar_n = {norm(v) for v in AR.values()}


def categoria(f):
    if f in gas_n: return 'Gás'
    if f in ar_n: return 'Ar de combustão'
    if f.startswith('clima_'): return 'Clima'
    if f.startswith('cal_') or f == 'biomassa_operando': return 'Calendário/regime'
    if f.startswith('y_'): return 'Histórico da saída'
    return 'Processo (P, T, nível, análise)'


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'MAE': mean_absolute_error(y, p), 'WAPE_%': 100 * np.abs(y - p).sum() / np.abs(y).sum(),
            'R2': r2_score(y, p)}


def novo_modelo():
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('rf', RandomForestRegressor(**RF_PARAMS))])


def filtrar_colunas(X_tr):
    ok = X_tr.columns[X_tr.isna().mean() <= MAX_FRAC_NAN_ENTRADA]
    return [c for c in ok if X_tr[c].nunique(dropna=True) > 1]


def avaliar(X, y, gap, baselines=None, nome=''):
    """Validação em janela deslizante + ajuste final e importância no último período."""
    m = y.notna()
    X, y = X[m], y[m]
    bl = {k: v[m] for k, v in (baselines or {}).items()}
    tss = TimeSeriesSplit(n_splits=N_SPLITS, gap=gap)
    linhas = []
    for dobra, (tr, te) in enumerate(tss.split(X)):
        cols = filtrar_colunas(X.iloc[tr])
        mod = novo_modelo().fit(X.iloc[tr][cols], y.iloc[tr])
        linhas.append({'dobra': dobra, 'modelo': 'Random Forest', **metricas(y.iloc[te], mod.predict(X.iloc[te][cols]))})
        for bn, bv in bl.items():
            ok = bv.iloc[te].notna()
            if ok.sum():
                linhas.append({'dobra': dobra, 'modelo': bn, **metricas(y.iloc[te][ok], bv.iloc[te][ok])})
    cv = pd.DataFrame(linhas)

    # Ajuste final: treino = início até (fim - teste - gap); teste = último FRAC_TESTE
    n_te = int(len(X) * FRAC_TESTE)
    tr_idx, te_idx = X.index[:len(X) - n_te - gap], X.index[len(X) - n_te:]
    cols = filtrar_colunas(X.loc[tr_idx])
    mod = novo_modelo().fit(X.loc[tr_idx, cols], y.loc[tr_idx])
    pred = pd.Series(mod.predict(X.loc[te_idx, cols]), index=te_idx)
    perm = permutation_importance(mod, X.loc[te_idx, cols], y.loc[te_idx], n_repeats=5,
                                  random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')
    imp = pd.DataFrame({
        'feature': cols,
        'categoria': [categoria(c) for c in cols],
        'pearson_treino': X.loc[tr_idx, cols].corrwith(y.loc[tr_idx]).values,
        'spearman_treino': X.loc[tr_idx, cols].corrwith(y.loc[tr_idx], method='spearman').values,
        'gini': mod.named_steps['rf'].feature_importances_,
        'perm_MAE': perm.importances_mean,       # aumento do MAE ao embaralhar (unidade do alvo)
    }).sort_values('perm_MAE', ascending=False).reset_index(drop=True)
    final = {'alvo': nome, 'n_treino': len(tr_idx), 'n_teste': len(te_idx),
             'teste_de': te_idx.min().date(), 'teste_ate': te_idx.max().date(),
             **metricas(y.loc[te_idx], pred)}
    return cv, final, imp, y.loc[te_idx], pred


def plot_resultado(nome, y_te, pred, imp, top=15):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1]})
    axes[0].plot(y_te.index, y_te, label='real', lw=1)
    axes[0].plot(pred.index, pred, label='RF', lw=1)
    if INICIO_BIOMASSA is not None and y_te.index.min() <= INICIO_BIOMASSA <= y_te.index.max():
        axes[0].axvline(INICIO_BIOMASSA, color='k', ls='--', lw=1)
    axes[0].set_title(f'{nome} — teste final')
    axes[0].legend()
    t = imp.head(top).iloc[::-1]
    cores = dict(zip(sorted(imp['categoria'].unique()), sns.color_palette('tab10')))
    axes[1].barh(t['feature'], t['perm_MAE'], color=[cores[c] for c in t['categoria']])
    axes[1].set_title('Permutation importance (↑ MAE)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=cores[c]) for c in cores]
    axes[1].legend(handles, cores.keys(), fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modo A — Processo: vapor de cada caldeira a gás a partir das entradas do mesmo dia
# MAGIC Esperado: **gás e ar da própria caldeira** dominando a importância (é a física da combustão).
# MAGIC Um R² alto aqui valida as tags e dá a relação gás → vapor usada na estimativa de gás evitado.

# COMMAND ----------

# DBTITLE 1,Modo A — treino e avaliação
cal = pd.DataFrame({'cal_dow': X_base.index.dayofweek, 'cal_mes': X_base.index.month,
                    'cal_fds': (X_base.index.dayofweek >= 5).astype(int)}, index=X_base.index)
XA = X_base.join(cal)

res_A, cv_A, imp_A = [], [], {}
for alvo in [a for a in ALVOS_MODO_A if a in Y]:
    base = {'Média do treino (referência)': pd.Series(Y[alvo].expanding().mean().shift(1), index=Y.index)}
    cv, final, imp, y_te, pred = avaliar(XA, Y[alvo], gap=0, baselines=base, nome=alvo)
    cv['alvo'] = alvo
    cv_A.append(cv); res_A.append(final); imp_A[alvo] = imp
    plot_resultado(f'Modo A — {alvo}', y_te, pred, imp)

print("=== Modo A — teste final ===")
display(pd.DataFrame(res_A).round(3))
print("=== Modo A — validação em janela deslizante (média das dobras) ===")
display(pd.concat(cv_A).groupby(['alvo', 'modelo'])[['MAE', 'WAPE_%', 'R2']].mean().reset_index().round(3))
for alvo, imp in imp_A.items():
    print(f"\n--- {alvo}: top 10 entradas ---")
    display(imp.head(10).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modo B — Previsão da demanda total (D+1, semana, mês)
# MAGIC Estratégia **direta**: um modelo por horizonte. No dia *t* só se usa informação até *t*:
# MAGIC entradas do dia *t*, histórico da demanda (defasagens e médias móveis) e o calendário do período previsto.
# MAGIC O *gap* entre treino e teste é igual ao horizonte, para que alvos de treino não enxerguem o período de teste.

# COMMAND ----------

# DBTITLE 1,Modo B — montar alvos futuros, entradas e linhas de base
dem = Y['demanda_total']

hist = pd.DataFrame({
    'y_lag0': dem, 'y_lag1': dem.shift(1), 'y_lag6': dem.shift(6), 'y_lag13': dem.shift(13),
    'y_mm7': dem.rolling(7, min_periods=5).mean(), 'y_mm30': dem.rolling(30, min_periods=20).mean(),
    'y_dp7': dem.rolling(7, min_periods=5).std(),
})


def alvo_futuro(H):
    """Soma de t+1 ... t+H (NaN se faltar algum dia)."""
    return dem[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1)


def calendario_futuro(H):
    idx = dem.index
    fds = pd.Series((idx.dayofweek >= 5).astype(int), index=idx)
    return pd.DataFrame({
        'cal_dow_prox': ((idx.dayofweek + 1) % 7),
        'cal_mes_prox': (idx + pd.Timedelta(days=1)).month,
        'cal_fds_no_periodo': fds[::-1].rolling(H, min_periods=1).sum()[::-1].shift(-1).values,
    }, index=idx)


def baselines(H):
    return {
        'Persistência (hoje × H)': dem * H,
        'Sazonal semanal': (dem.shift(6) if H == 1 else dem.rolling(7).sum() * (H / 7)),
        'Média 7 dias × H': dem.rolling(7).mean() * H,
    }

# COMMAND ----------

# DBTITLE 1,Modo B — treino e avaliação por horizonte
res_B, cv_B, imp_B = [], [], {}
for nome_h, H in HORIZONTES.items():
    XB = X_base.join(hist).join(calendario_futuro(H))
    yB = alvo_futuro(H)
    cv, final, imp, y_te, pred = avaliar(XB, yB, gap=H, baselines=baselines(H), nome=nome_h)
    cv['horizonte'] = nome_h
    cv_B.append(cv); res_B.append(final); imp_B[nome_h] = imp
    plot_resultado(f'Modo B — {nome_h}', y_te, pred, imp)

print("=== Modo B — teste final (Random Forest) ===")
display(pd.DataFrame(res_B).round(3))

cvB = pd.concat(cv_B).groupby(['horizonte', 'modelo'])[['MAE', 'WAPE_%', 'R2']].mean().reset_index()
print("=== Modo B — janela deslizante: RF × linhas de base (média das dobras) ===")
display(cvB.round(3))

# Ganho do RF sobre a melhor linha de base (WAPE)
ganho = []
for hz, g_ in cvB.groupby('horizonte'):
    rf = g_.loc[g_['modelo'] == 'Random Forest', 'WAPE_%'].iloc[0]
    melhor = g_[g_['modelo'] != 'Random Forest'].sort_values('WAPE_%').iloc[0]
    ganho.append({'horizonte': hz, 'WAPE_RF_%': rf, 'melhor_baseline': melhor['modelo'],
                  'WAPE_baseline_%': melhor['WAPE_%'], 'RF_melhor?': rf < melhor['WAPE_%']})
print("=== O RF supera a melhor linha de base? ===")
display(pd.DataFrame(ganho).round(2))

# COMMAND ----------

# DBTITLE 1,Importância por categoria de entrada (modos A e B)
linhas = []
for modo, dic in [('A', imp_A), ('B', imp_B)]:
    for alvo, imp in dic.items():
        tot = imp['perm_MAE'].clip(lower=0).sum()
        for cat, v in imp.groupby('categoria')['perm_MAE'].apply(lambda s: s.clip(lower=0).sum()).items():
            linhas.append({'modo': modo, 'alvo': alvo, 'categoria': cat, 'participacao_%': 100 * v / tot if tot else np.nan})
cat_df = pd.DataFrame(linhas).pivot_table(index=['modo', 'alvo'], columns='categoria', values='participacao_%').fillna(0)
display(cat_df.reset_index().round(1))

ax = cat_df.plot(kind='barh', stacked=True, figsize=(12, 0.6 * len(cat_df) + 2), colormap='tab10')
ax.set_xlabel('% da permutation importance')
ax.set_title('De onde vem a capacidade preditiva, por categoria de entrada')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Salvar resultados no 3_gold (arquivos novos)
os.makedirs(RESULT_DIR, exist_ok=True)
pd.DataFrame(res_A).to_csv(f"{RESULT_DIR}/rf_vapor_modoA_teste.csv", index=False)
pd.DataFrame(res_B).to_csv(f"{RESULT_DIR}/rf_vapor_modoB_teste.csv", index=False)
cvB.to_csv(f"{RESULT_DIR}/rf_vapor_modoB_cv.csv", index=False)
pd.concat([i.assign(modo='A', alvo=a) for a, i in imp_A.items()] +
          [i.assign(modo='B', alvo=a) for a, i in imp_B.items()]).to_csv(
    f"{RESULT_DIR}/rf_vapor_importancias.csv", index=False)
print(f"Resultados salvos em {RESULT_DIR}")