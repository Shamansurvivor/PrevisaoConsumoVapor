# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Random Forest diário — explicação por caldeira (Modo A) e previsão por horizonte (Modo B) — rev2
# MAGIC
# MAGIC - **Modo A** — explica a geração diária de cada caldeira a gás (C208, C502, C550) a partir das entradas do
# MAGIC   mesmo dia (inclusive gás e ar). Não é previsão: mede quanto cada grupo de variáveis explica a saída.
# MAGIC - **Modo B** — prevê a demanda total de vapor para **D+1, 7 dias e 30 dias**, usando só o que se conhece no
# MAGIC   fim do dia da previsão, e compara o RF com linhas de base (persistência, sazonal semanal, média de 7 dias).
# MAGIC
# MAGIC **rev2 — o que mudou em relação ao rev1**
# MAGIC - Fonte de dados: lê `workspace.previsao_vapor.silver_prepared` (gerada por `random_forest_prep_rev3`) em vez de
# MAGIC   `bronze_processo` + `bronze_clima`. Mesma base do `rf_training_multi_target_rev3`.
# MAGIC - Período configurável (`DATA_INICIO = 2023-01-01`, `DATA_FIM = None` = último dado) + trava contra tabela desatualizada.
# MAGIC - Da `silver_prepared` usa só as medições horárias das tags e o clima; descarta as features criadas no prep
# MAGIC   (lags, médias móveis, calendário horário, O₂ defasado) e os totalizadores `_TOT`, porque este notebook
# MAGIC   cria as próprias features diárias.
# MAGIC - **Gráficos de previsão D+1, semana+1 e mês+1** com todos os algoritmos do `rf_training_multi_target_rev3`
# MAGIC   (RF horário recursivo, RF diário, Prophet, SARIMA e ensembles), lidos da tabela `gold_ensemble_forecast_30d`,
# MAGIC   mais o RF direto do Modo B deste notebook.
# MAGIC
# MAGIC **Ordem:** `random_forest_prep_rev3` → `rf_training_multi_target_rev3` → este notebook.

# COMMAND ----------

# DBTITLE 1,Imports
# MAGIC %pip install seaborn --quiet
# MAGIC import os
# MAGIC import re
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.impute import SimpleImputer
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.model_selection import TimeSeriesSplit
# MAGIC from sklearn.inspection import permutation_importance
# MAGIC from sklearn.metrics import r2_score, mean_absolute_error
# MAGIC
# MAGIC sns.set_style("whitegrid")
# MAGIC pd.set_option('display.max_columns', None)
# MAGIC pd.set_option('display.width', 200)

# COMMAND ----------

# DBTITLE 1,Install missing package for imports
# MAGIC %pip install seaborn --quiet

# COMMAND ----------

# DBTITLE 1,Configuração
# --- Fonte e período ---
SILVER_TABLE = "workspace.previsao_vapor.silver_prepared"
ENSEMBLE_TABLE = "workspace.previsao_vapor.gold_ensemble_forecast_30d"   # previsões do rf_training_multi_target_rev3
DATA_INICIO = pd.Timestamp("2023-01-01 00:00:00")
DATA_FIM = None          # None = até o último dado coletado

# Colunas de clima na silver_prepared
CLIMA_COLS = ['temperature_C', 'relative_humidity_pct', 'surface_pressure_hPa', 'pressure_msl_hPa']
# Features horárias criadas no prep (descartadas aqui: este notebook cria as próprias features diárias)
CAL_COLS_PREP = ['hour', 'dayofweek', 'month', 'dayofyear',
                 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

# SAÍDAS do sistema (vapor gerado)
SAIDAS = {'C208': '11FI208-02', 'C502': '11FI502-30', 'C550': '21FI550-30', 'BIO': '21FI551-10'}

# Entradas de combustão (só para agrupar a importância por categoria)
GAS = {'C208': '11FC208-106.PV', 'C502': '11FC502-01', 'C550': '21FC550-72'}
AR = {'C208': '11FC208-138', 'C502': '11FC502-04', 'C550': '21FC550-69'}

MIN_HORAS_DIA = 20            # dia válido com >= 20 h de dado
MAX_FRAC_NAN_ENTRADA = 0.5    # entrada descartada se > 50% faltante no treino
FRAC_BIOMASSA_ON = 0.05       # biomassa "operando" acima de 5% do seu p95 diário

HORIZONTES = {'D+1': 1, 'Semana (7d)': 7, 'Mês (30d)': 30}   # modo B
ALVOS_MODO_A = ['C208', 'C502', 'C550']                      # modo A

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1)
N_SPLITS = 5          # dobras da validação em janela deslizante
FRAC_TESTE = 0.2      # teste final (último período) para importância

print(f"Fonte: {SILVER_TABLE} | período: {DATA_INICIO} -> {'último dado coletado' if DATA_FIM is None else DATA_FIM}")

# COMMAND ----------

# DBTITLE 1,Carregar dados da silver_prepared
# ============================================================
# Lê a mesma tabela do rf_training_multi_target_rev3 e reconstrói:
#   h       -> medições horárias das tags (sem _TOT e sem features derivadas)
#   d_media -> médias diárias (dia válido com >= MIN_HORAS_DIA h) + clima diário (clima_*)
# ============================================================
norm = lambda t: t.upper().replace('.PV', '').strip()

df_s = spark.table(SILVER_TABLE).toPandas()
df_s['datetime'] = pd.to_datetime(df_s['datetime'])
df_s = df_s.set_index('datetime').sort_index()
df_s = df_s.loc[DATA_INICIO:] if DATA_FIM is None else df_s.loc[DATA_INICIO:DATA_FIM]
df_s = df_s[~df_s.index.duplicated(keep='first')]

assert df_s.index.min() < DATA_INICIO + pd.Timedelta(days=7), (
    f"{SILVER_TABLE} começa em {df_s.index.min()} — rode random_forest_prep_rev3 antes deste notebook."
)

# Separar o que é medição de tag do que foi criado no prep
derivadas = [c for c in df_s.columns if '_lag_' in c or '_ma_' in c or c in CAL_COLS_PREP]
totais = [c for c in df_s.columns if '_TOT' in c.upper()]
clima_cols = [c for c in CLIMA_COLS if c in df_s.columns]
tags = [c for c in df_s.columns if c not in set(derivadas) | set(totais) | set(clima_cols)]

h = df_s[tags].copy()
h.columns = [norm(c) for c in h.columns]
h = h.loc[:, ~h.columns.duplicated()].asfreq('h')
print(f"silver_prepared: {df_s.shape} | {df_s.index.min()} a {df_s.index.max()}")
print(f"Tags horárias usadas: {len(tags)} | descartadas: {len(derivadas)} derivadas do prep, {len(totais)} _TOT")
print(f"Horário: {h.shape} | {h.index.min()} a {h.index.max()}")

# Conferir que as saídas existem
faltam = [f"{k}={v}" for k, v in SAIDAS.items() if norm(v) not in h]
if faltam:
    print(">>> ATENÇÃO — saídas não encontradas na silver_prepared:", faltam)

# Agregação diária: média do dia, válida só com >= MIN_HORAS_DIA horas
g = h.resample('D')
d_media = g.mean().where(g.count() >= MIN_HORAS_DIA)

# Clima (média diária)
if clima_cols:
    cl = df_s[clima_cols].resample('D').mean().add_prefix('clima_')
    d_media = d_media.join(cl, how='left')
    print(f"Clima: {list(cl.columns)} | dias sem clima: {int(cl.isna().all(axis=1).sum())}")
else:
    print("Clima não encontrado na silver_prepared.")

print(f"Diário: {d_media.shape} | {d_media.index.min().date()} a {d_media.index.max().date()}")

# COMMAND ----------

# DBTITLE 1,Saídas diárias, partida da biomassa e entradas
Y = pd.DataFrame(index=d_media.index)
for k, tag in SAIDAS.items():
    if norm(tag) in d_media:
        Y[k] = d_media[norm(tag)] * 24   # total do dia

# Partida da biomassa e demanda total (biomassa = 0 antes da partida)
INICIO_BIOMASSA = None
if 'BIO' in Y and Y['BIO'].notna().any():
    lim = FRAC_BIOMASSA_ON * Y['BIO'].quantile(0.95)
    INICIO_BIOMASSA = Y.index[(Y['BIO'] > lim).fillna(False)].min()
    Y.loc[Y.index < INICIO_BIOMASSA, 'BIO'] = 0.0
    print(f"Início da biomassa detectado: {INICIO_BIOMASSA}")
# INICIO_BIOMASSA = pd.Timestamp('2026-05-01')   # descomente para fixar manualmente

gas_boilers = [c for c in ['C208', 'C502', 'C550'] if c in Y]
Y['demanda_total'] = Y[gas_boilers + (['BIO'] if 'BIO' in Y else [])].sum(
    axis=1, min_count=len(gas_boilers) + ('BIO' in Y))

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

# DBTITLE 1,Funções: categorias, métricas, modelo e validação
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
        linhas.append({'dobra': dobra, 'modelo': 'Random Forest',
                       **metricas(y.iloc[te], mod.predict(X.iloc[te][cols]))})
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
        'perm_MAE': perm.importances_mean,   # aumento do MAE ao embaralhar (unidade do alvo)
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
# MAGIC ## Modo A — o que explica a geração diária de cada caldeira a gás
# MAGIC Entradas do **mesmo dia** (inclusive gás e ar de combustão) + calendário. Serve para medir a contribuição de
# MAGIC cada grupo de variáveis; não é um modelo de previsão.

# COMMAND ----------

# DBTITLE 1,Modo A — treino e avaliação por caldeira
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
# MAGIC ## Modo B — previsão da demanda total para D+1, 7 dias e 30 dias
# MAGIC Alvo = soma da demanda de t+1 até t+H. Entradas = o que se conhece no fim do dia t (medições do dia,
# MAGIC histórico da demanda, calendário dos dias à frente). O RF é comparado com linhas de base simples.

# COMMAND ----------

# DBTITLE 1,Modo B — histórico, alvo futuro, calendário e linhas de base
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
res_B, cv_B, imp_B, prev_modoB = [], [], {}, {}
for nome_h, H in HORIZONTES.items():
    XB = X_base.join(hist).join(calendario_futuro(H))
    yB = alvo_futuro(H)
    cv, final, imp, y_te, pred = avaliar(XB, yB, gap=H, baselines=baselines(H), nome=nome_h)
    cv['horizonte'] = nome_h
    cv_B.append(cv); res_B.append(final); imp_B[nome_h] = imp
    plot_resultado(f'Modo B — {nome_h}', y_te, pred, imp)

    # Previsão real: re-treina com todo o histórico e prevê a partir do último dia com demanda válida
    m_ok = yB.notna()
    cols_f = filtrar_colunas(XB[m_ok])
    mod_f = novo_modelo().fit(XB.loc[m_ok, cols_f], yB[m_ok])
    dia_base = dem.last_valid_index()
    prev_modoB[nome_h] = {'dia_base': dia_base,
                          'previsao': float(mod_f.predict(XB.loc[[dia_base], cols_f])[0])}

print("=== Modo B — previsão do RF direto a partir do último dia ===")
for nome_h, v in prev_modoB.items():
    print(f"  {nome_h:12s}: base {v['dia_base'].date()} -> {v['previsao']:.1f}")

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

# MAGIC %md
# MAGIC ## Previsão do consumo de vapor — D+1, semana+1 e mês+1
# MAGIC Algoritmos do `rf_training_multi_target_rev3` (tabela `gold_ensemble_forecast_30d`):
# MAGIC RF horário recursivo, RF diário, Prophet, SARIMA e os ensembles (média, mediana e ponderado),
# MAGIC mais o **RF direto (Modo B)** deste notebook, que prevê diretamente a soma de cada horizonte.

# COMMAND ----------

# DBTITLE 1,Carregar previsões dos algoritmos (rf_training_multi_target_rev3)
ALGOS = {
    'RF_Horario': 'RF horário recursivo',
    'RF_Diario': 'RF diário',
    'Prophet': 'Prophet',
    'SARIMA': 'SARIMA',
    'Ensemble_Media': 'Ensemble — média',
    'Ensemble_Mediana': 'Ensemble — mediana',
    'Ensemble_Ponderado': 'Ensemble — ponderado',
}
CORES = {'RF_Horario': '#FF9800', 'RF_Diario': '#2196F3', 'Prophet': '#4CAF50', 'SARIMA': '#9C27B0',
         'Ensemble_Media': '#E91E63', 'Ensemble_Mediana': '#00BCD4', 'Ensemble_Ponderado': '#000000',
         'RF_ModoB': '#795548'}
HZ = {'D+1': 1, 'Semana+1': 7, 'Mês+1': 30}
MAPA_MODOB = {'D+1': 'D+1', 'Semana+1': 'Semana (7d)', 'Mês+1': 'Mês (30d)'}

ens = spark.table(ENSEMBLE_TABLE).toPandas()
ens['data'] = pd.to_datetime(ens['data'])
ens = ens.set_index('data').sort_index()
algos_ok = [a for a in ALGOS if a in ens.columns]

dado_ate_ens = pd.Timestamp(ens['dado_ate'].iloc[0]) if 'dado_ate' in ens else None
print(f"Previsões de {ens.index.min().date()} a {ens.index.max().date()} | algoritmos: {algos_ok}")
print(f"Treino dos algoritmos com dado até: {dado_ate_ens} | silver_prepared até: {df_s.index.max()}")
if dado_ate_ens is not None and dado_ate_ens.normalize() != df_s.index.max().normalize():
    print(">>> ATENÇÃO: a tabela de previsões não é da mesma carga da silver_prepared. "
          "Rode o rf_training_multi_target_rev3 de novo.")

real_dia = Y['demanda_total']   # consumo diário real (mesma unidade: soma do dia)

def resumo_horizonte(nome_h):
    """Soma prevista no horizonte por algoritmo (+ RF direto do Modo B) e IC 95% do ensemble."""
    H = HZ[nome_h]
    jan = ens.iloc[:H]
    tot = {a: jan[a].sum() for a in algos_ok}
    if MAPA_MODOB.get(nome_h) in prev_modoB:
        tot['RF_ModoB'] = prev_modoB[MAPA_MODOB[nome_h]]['previsao']
    ic = (jan['IC_lower'].sum(), jan['IC_upper'].sum()) if {'IC_lower', 'IC_upper'} <= set(jan.columns) else None
    return jan, pd.Series(tot), ic

def rotulo(a):
    return ALGOS.get(a, 'RF direto (Modo B)')

# COMMAND ----------

# DBTITLE 1,Gráfico — previsão D+1
jan, tot, ic = resumo_horizonte('D+1')
dia = jan.index[0]
ult7 = real_dia.dropna().tail(7)

fig, ax = plt.subplots(figsize=(12, 5))
ordem = tot.sort_values().index
ax.barh([rotulo(a) for a in ordem], tot[ordem].values, color=[CORES[a] for a in ordem])
for y_, v in enumerate(tot[ordem].values):
    ax.text(v, y_, f' {v:,.0f}', va='center', fontsize=9)
ax.axvline(ult7.iloc[-1], color='gray', ls='-', lw=1.2, label=f'real do último dia ({ult7.index[-1].date()}): {ult7.iloc[-1]:,.0f}')
ax.axvline(ult7.mean(), color='red', ls='--', lw=1.2, label=f'média real dos últimos 7 dias: {ult7.mean():,.0f}')
if ic is not None:
    ax.axvspan(ic[0], ic[1], color='gray', alpha=0.08, label=f'IC 95% (Prophet + SARIMA): {ic[0]:,.0f} – {ic[1]:,.0f}')
ax.set_xlim(min(tot.min(), ult7.min()) * 0.9, max(tot.max(), ult7.max()) * 1.08)
ax.set_xlabel('Consumo de vapor no dia (soma das 24 h)')
ax.set_title(f'Previsão D+1 — consumo de vapor em {dia:%d/%m/%Y}, por algoritmo', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.show()

display(tot.rename(index=rotulo).rename('previsao_D+1').round(1).to_frame())

# COMMAND ----------

# DBTITLE 1,Gráfico — previsão semana+1
def grafico_horizonte(nome_h, dias_hist):
    jan, tot, ic = resumo_horizonte(nome_h)
    hist = real_dia.dropna().tail(dias_hist)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})

    # Esquerda: histórico real + trajetória diária prevista por algoritmo
    ax = axes[0]
    ax.plot(hist.index, hist.values, color='black', lw=1.6, label='Real')
    for a in algos_ok:
        estilo = dict(lw=2.6) if a == 'Ensemble_Ponderado' else dict(lw=1.2, alpha=0.85)
        ax.plot(jan.index, jan[a].values, color=CORES[a], label=ALGOS[a], **estilo)
    if {'IC_lower', 'IC_upper'} <= set(jan.columns):
        ax.fill_between(jan.index, jan['IC_lower'], jan['IC_upper'], color='gray', alpha=0.12, label='IC 95%')
    ax.axvline(jan.index[0] - pd.Timedelta(hours=12), color='gray', ls=':', lw=1)
    ax.set_ylabel('Consumo de vapor por dia')
    ax.set_title(f'{nome_h}: últimos {dias_hist} dias reais + previsão diária '
                 f'({jan.index[0]:%d/%m} a {jan.index[-1]:%d/%m/%Y})')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.tick_params(axis='x', rotation=30)

    # Direita: total do horizonte por algoritmo
    ax = axes[1]
    ordem = tot.sort_values().index
    ax.barh([rotulo(a) for a in ordem], tot[ordem].values, color=[CORES[a] for a in ordem])
    for y_, v in enumerate(tot[ordem].values):
        ax.text(v, y_, f' {v:,.0f}', va='center', fontsize=8)
    ref = hist.tail(HZ[nome_h]).sum() if len(hist) >= HZ[nome_h] else np.nan
    if pd.notna(ref):
        ax.axvline(ref, color='red', ls='--', lw=1.2, label=f'real dos últimos {HZ[nome_h]} dias: {ref:,.0f}')
    if ic is not None:
        ax.axvspan(ic[0], ic[1], color='gray', alpha=0.08, label='IC 95% (soma)')
    ax.set_xlim(tot.min() * 0.9, max(tot.max(), ref if pd.notna(ref) else 0) * 1.1)
    ax.set_title(f'Total previsto — {nome_h} ({HZ[nome_h]} dias)')
    ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(f'Previsão {nome_h} do consumo de vapor — todos os algoritmos', fontweight='bold')
    plt.tight_layout()
    plt.show()
    display(tot.rename(index=rotulo).rename(f'previsao_{nome_h}').round(1).to_frame())
    return tot

tot_semana = grafico_horizonte('Semana+1', dias_hist=28)

# COMMAND ----------

# DBTITLE 1,Gráfico — previsão mês+1
tot_mes = grafico_horizonte('Mês+1', dias_hist=90)

# COMMAND ----------

# DBTITLE 1,Tabela-resumo dos três horizontes
resumo_hz = pd.DataFrame({h: resumo_horizonte(h)[1] for h in HZ})
resumo_hz.index = [rotulo(a) for a in resumo_hz.index]
resumo_hz.index.name = 'algoritmo'
print(f"Consumo de vapor previsto (soma no horizonte) a partir de {ens.index[0].date()}")
display(resumo_hz.round(1))

# COMMAND ----------

# DBTITLE 1,Importância por categoria de entrada (Modos A e B)
linhas = []
for modo, dic in [('A', imp_A), ('B', imp_B)]:
    for alvo, imp in dic.items():
        tot = imp['perm_MAE'].clip(lower=0).sum()
        for cat, v in imp.groupby('categoria')['perm_MAE'].apply(lambda s: s.clip(lower=0).sum()).items():
            linhas.append({'modo': modo, 'alvo': alvo, 'categoria': cat,
                           'participacao_%': 100 * v / tot if tot else np.nan})
cat_df = pd.DataFrame(linhas).pivot_table(index=['modo', 'alvo'], columns='categoria',
                                          values='participacao_%').fillna(0)
display(cat_df.reset_index().round(1))

ax = cat_df.plot(kind='barh', stacked=True, figsize=(12, 0.6 * len(cat_df) + 2), colormap='tab10')
ax.set_xlabel('% da permutation importance')
ax.set_title('De onde vem a capacidade preditiva, por categoria de entrada')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Salvar resultados (Delta)
SCHEMA = "workspace.previsao_vapor"

def salvar(pdf, nome):
    pdf = pdf.copy()
    for c in pdf.columns:   # datas (teste_de/teste_ate) como texto para o Spark
        if pdf[c].dtype == object:
            pdf[c] = pdf[c].astype(str)
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{nome}"))

salvar(pd.DataFrame(res_A), "gold_test_modoA")
salvar(pd.DataFrame(res_B), "gold_test_modoB")
salvar(cvB.reset_index(drop=True), "gold_cv_modoB")
imp_df = pd.concat([i.assign(modo='A', alvo=a) for a, i in imp_A.items()] +
                   [i.assign(modo='B', alvo=a) for a, i in imp_B.items()])
salvar(imp_df, "gold_importancias")
salvar(resumo_hz.reset_index(), "gold_vapor_previsao_horizontes")

print(f"Resultados salvos em Delta tables: {SCHEMA}.gold_test_modoA, gold_test_modoB, gold_cv_modoB, "
      f"gold_importancias, gold_vapor_previsao_horizontes")

# COMMAND ----------

# DBTITLE 1,Diagnóstico de uma tag (exemplo: gás da C208)
tag = '11FC208-106'
s_h = h[tag]
print(f"Horário: {s_h.notna().mean():.1%} das horas com dado | de {s_h.first_valid_index()} até {s_h.last_valid_index()}")
print(f"Diário : {d_media[tag].notna().mean():.1%} dos dias válidos (>= {MIN_HORAS_DIA} h de dado)")
print(s_h.describe().round(2))

imp = imp_A['C208']
linha = imp[imp['feature'] == tag]
if len(linha):
    print(f"\nEntrou no modelo da C208 — posição {linha.index[0] + 1} de {len(imp)} no ranking")
    display(linha.round(4))
else:
    print("\nNÃO entrou no modelo da C208 (descartada pelo filtro de lacunas ou por ser constante)")