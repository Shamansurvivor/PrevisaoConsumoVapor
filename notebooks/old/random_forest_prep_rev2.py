# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
# ]
# ///
# MAGIC %md
# MAGIC # Preparação de dados para Random Forest — Previsão de Consumo de Vapor (rev2)
# MAGIC
# MAGIC Carrega os dados de processo (`bronze_processo`) e climáticos (`bronze_clima`), faz o merge em uma tabela
# MAGIC horária unificada, realiza EDA, cria features sem vazamento e grava `workspace.previsao_vapor.silver_prepared`
# MAGIC para o notebook `rf_training_multi_target_rev1`.
# MAGIC
# MAGIC **rev2 — o que mudou em relação ao rev1**
# MAGIC - Período configurável: `DATA_INICIO = 2023-01-01` e `DATA_FIM = None` (até o último dado coletado).
# MAGIC - Filtro de período aplicado no Spark, antes da agregação horária.
# MAGIC - Merge processo + clima passou de **INNER** para **LEFT**: o fim do período é o último dado de processo,
# MAGIC   não o fim do `bronze_clima` (30/06/2026).
# MAGIC - Complemento opcional do clima faltante via Open-Meteo Archive (preencher `LAT`/`LON`).
# MAGIC - Relatório de cobertura por ano para conferir o período 2023–2026.

# COMMAND ----------

# DBTITLE 1,Instalar seaborn
# MAGIC %pip install seaborn

# COMMAND ----------

# DBTITLE 1,Imports
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("Bibliotecas importadas com sucesso.")

# COMMAND ----------

# DBTITLE 1,Configuração central
# ============================================================
# CONFIGURAÇÃO CENTRAL
# ============================================================
# Tudo que define período, targets e o que não pode ser feature
# fica aqui, para ser usado igual em TODAS as células seguintes.
# ============================================================

# --- Período de dados ---
DATA_INICIO = pd.Timestamp("2023-01-01 00:00:00")
DATA_FIM = None          # None = até o último dado coletado

# --- Complemento climático (Open-Meteo Archive) ---
# Preencha com as MESMAS coordenadas usadas na carga original do bronze_clima.
# Com None, as horas sem clima ficam NaN e são imputadas no modelo.
LAT, LON = None, None
SALVAR_CLIMA_NOVO_NO_BRONZE = False   # True = acrescenta as horas novas ao bronze_clima

# --- Targets e exclusões ---
MULTI_TARGETS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']

# Gás e ar de combustão: são consequência da demanda de vapor no mesmo
# instante (a caldeira queima para atender o consumo) -> proxy do target.
EXCLUDE_FEATURES = ['11FC550-72.PV', '11FC208-106.PV']

# Horizonte de previsão em horas. Com HORIZON = 1 prevê-se a próxima hora.
# Lags/médias móveis só podem usar informação disponível até t - HORIZON.
HORIZON = 1
LAGS = [l for l in [1, 3, 6, 12, 24] if l >= HORIZON]
MA_WINDOWS = [3, 6, 12, 24]

RANDOM_STATE = 42

def is_target_derived(col):
    """Totalizadores (_TOT, _TOT_DAY) são somas acumuladas de vazão -> nunca entram como feature."""
    return '_TOT' in col

print(f"Período: {DATA_INICIO} -> {'último dado coletado' if DATA_FIM is None else DATA_FIM}")
print(f"Targets: {MULTI_TARGETS}")
print(f"Excluídas (gás/ar): {EXCLUDE_FEATURES}")
print(f"Horizonte: {HORIZON} h | Lags: {LAGS} | Janelas MA: {MA_WINDOWS}")

# COMMAND ----------

# DBTITLE 1,Carregar dados de processo (Delta table)
# ============================================================
# Lê bronze_processo (formato long: tag, ts, value), filtra o período,
# agrega para frequência horária no Spark e faz pivot para wide.
# ============================================================

from pyspark.sql.functions import col, date_trunc, avg

bronze_spark = (spark.table("workspace.previsao_vapor.bronze_processo")
                .filter(col("ts") >= DATA_INICIO))
if DATA_FIM is not None:
    bronze_spark = bronze_spark.filter(col("ts") <= DATA_FIM)

hourly = (
    bronze_spark
    .withColumn("hour", date_trunc("hour", col("ts")))
    .groupBy("hour", "tag")
    .agg(avg("value").alias("value"))
)

print("Pivoting horário...")
hourly_wide = hourly.groupBy("hour").pivot("tag").agg(avg("value"))

print("Convertendo para pandas...")
df_process = hourly_wide.toPandas()
df_process = df_process.set_index(pd.to_datetime(df_process["hour"])).drop(columns=["hour"])
df_process = df_process.sort_index()
df_process.index.name = 'datetime'

# all_series (compatibilidade com células que ainda referenciam)
all_series = {}
for tag in df_process.columns:
    s = df_process[tag].dropna()
    all_series[tag] = pd.DataFrame({'ts': s.index, 'value': s.values})

print(f"Tabela wide de processo (horária): {df_process.shape}")
print(f"Período: {df_process.index.min()} a {df_process.index.max()}")
print(f"Tags carregadas: {len(all_series)}")

print("\n=== Resumo das primeiras tags ===")
for tag in list(all_series.keys())[:5]:
    print(f"  {tag}: {len(all_series[tag]):,} registros | {all_series[tag]['ts'].min()} a {all_series[tag]['ts'].max()}")
print(f"  ... e mais {len(all_series)-5} tags")

display(df_process.head(3))

# COMMAND ----------

# DBTITLE 1,Carregar dados climáticos + complemento + merge
# ============================================================
# 1. Lê bronze_clima
# 2. Se o clima terminar antes do processo, completa via Open-Meteo
#    Archive (se LAT/LON preenchidos)
# 3. LEFT join a partir do processo + grade horária regular
# ============================================================

import requests

df_clima = spark.table("workspace.previsao_vapor.bronze_clima").toPandas()
df_clima['datetime_local'] = pd.to_datetime(df_clima['datetime_local'])
df_clima = df_clima.set_index('datetime_local')
df_clima.index.name = 'datetime'
df_clima = df_clima[~df_clima.index.duplicated(keep='first')].sort_index()

fim_proc, fim_clima = df_process.index.max(), df_clima.index.max()
print(f"Dados climáticos (bronze): {df_clima.shape}")
print(f"Período: {df_clima.index.min()} a {fim_clima}")

if fim_clima < fim_proc:
    if LAT is None or LON is None:
        print(f"\nATENÇÃO: clima termina em {fim_clima}, processo em {fim_proc}.")
        print("Sem LAT/LON, essas horas ficam com clima NaN (imputado pela média no modelo).")
    else:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": LAT, "longitude": LON,
                "start_date": (fim_clima + pd.Timedelta(hours=1)).date().isoformat(),
                "end_date": fim_proc.date().isoformat(),
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,pressure_msl",
                "timezone": "America/Sao_Paulo",
            },
            timeout=60,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        clima_novo = pd.DataFrame({
            "temperature_C": h["temperature_2m"],
            "relative_humidity_pct": h["relative_humidity_2m"],
            "surface_pressure_hPa": h["surface_pressure"],
            "pressure_msl_hPa": h["pressure_msl"],
        }, index=pd.to_datetime(h["time"]))
        clima_novo.index.name = 'datetime'
        clima_novo = clima_novo[clima_novo.index > fim_clima]
        clima_novo = clima_novo[[c for c in df_clima.columns if c in clima_novo.columns]]
        df_clima = pd.concat([df_clima, clima_novo]).sort_index()
        print(f"\nClima complementado (Open-Meteo): +{len(clima_novo):,} h -> até {df_clima.index.max()}")

        if SALVAR_CLIMA_NOVO_NO_BRONZE and len(clima_novo):
            (spark.createDataFrame(clima_novo.reset_index().rename(columns={'datetime': 'datetime_local'}))
                  .write.mode("append").option("mergeSchema", "true")
                  .saveAsTable("workspace.previsao_vapor.bronze_clima"))
            print("Horas novas acrescentadas ao bronze_clima.")

# LEFT join: o período é definido pelo processo (último dado coletado)
df_merged = df_process.join(df_clima, how='left')
df_merged = df_merged[~df_merged.index.duplicated(keep='first')].sort_index()

n_antes = len(df_merged)
# Grade horária regular: horas faltantes viram linhas NaN. Sem isso, shift(1)
# pode "pular" horas e o lag_1h deixa de ser 1 hora.
df_merged = df_merged.asfreq('h')
print(f"\nHoras reinseridas na grade regular: {len(df_merged) - n_antes:,}")

print(f"\n=== Tabela unificada (processo + clima) ===")
print(f"Shape: {df_merged.shape}")
print(f"Período: {df_merged.index.min()} a {df_merged.index.max()}")

# Cobertura por ano
tgts_presentes = [t for t in MULTI_TARGETS if t in df_merged.columns]
clima_cols_presentes = [c for c in df_clima.columns if c in df_merged.columns]
cobertura = pd.DataFrame({
    'horas': df_merged.groupby(df_merged.index.year).size(),
    'horas_com_target': df_merged[tgts_presentes].notna().any(axis=1).groupby(df_merged.index.year).sum(),
    'horas_com_clima': df_merged[clima_cols_presentes].notna().any(axis=1).groupby(df_merged.index.year).sum(),
})
cobertura.index.name = 'ano'
print("\n=== Cobertura por ano ===")
display(cobertura)
display(df_merged.head(3))

# COMMAND ----------

# DBTITLE 1,EDA — shape, describe, missing
print("=== SHAPE ===")
print(f"Linhas: {df_merged.shape[0]:,} | Colunas: {df_merged.shape[1]}")

print("\n=== DESCRIBE ===")
display(df_merged.describe().T)

print("\n=== MISSING VALUES ===")
missing = df_merged.isnull().sum()
missing_pct = (missing / len(df_merged) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print(f"Colunas com missing: {len(missing_df)} de {len(df_merged.columns)}")
if len(missing_df) > 0:
    display(missing_df)
else:
    print("Nenhum missing value encontrado.")

# COMMAND ----------

# DBTITLE 1,EDA — tipos de features
feature_types = []
for c in df_merged.columns:
    dtype = df_merged[c].dtype
    n_unique = df_merged[c].nunique()
    if dtype in ('float64', 'float32', 'int64', 'int32'):
        ftype = 'numeric ordinal' if n_unique <= 20 else 'numeric float'
    elif dtype == 'object':
        ftype = 'categorical/nominal'
    else:
        ftype = str(dtype)
    feature_types.append({'feature': c, 'type': ftype, 'n_unique': n_unique})

feature_type_df = pd.DataFrame(feature_types)
print("=== Tabela de tipos de features ===")
display(feature_type_df)

# COMMAND ----------

# DBTITLE 1,EDA — mapa de correlações
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
corr_matrix = df_merged[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 5})
ax.set_title('Mapa de Correlações — Variáveis de Processo e Clima', fontsize=14)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA — correlação das features com cada target
# Exclui gás/ar e _TOT da mesma forma que o RF. Análise EXPLORATÓRIA
# (período inteiro, sem lags). A comparação justa com o RF está na
# célula de importância.

multi_targets_valid = [t for t in MULTI_TARGETS if t in df_merged.columns]
print(f"Targets encontrados: {multi_targets_valid}")

numeric_cols = df_merged.select_dtypes(include=[np.number]).columns

feature_cols = [c for c in numeric_cols
                if c not in multi_targets_valid
                and c not in EXCLUDE_FEATURES
                and not is_target_derived(c)]

corr_dict = {tgt: df_merged[feature_cols].corrwith(df_merged[tgt]) for tgt in multi_targets_valid}
multi_corr_df = pd.DataFrame(corr_dict).dropna(how='all')
multi_corr_df.index.name = 'feature'
multi_corr_df['media_abs'] = multi_corr_df[multi_targets_valid].abs().mean(axis=1)
multi_corr_df = multi_corr_df.sort_values('media_abs', ascending=False)

print(f"\n=== Correlação de Pearson por Target (exploratória, sem gás/ar e sem _TOT) ===")
display(multi_corr_df.round(4))

high_corr = multi_corr_df[(multi_corr_df[multi_targets_valid].abs() > 0.5).any(axis=1)]
print(f"\nFeatures com |corr| > 0.5 em pelo menos um target: {len(high_corr)}")
display(high_corr.round(4))

top_n = min(15, len(multi_corr_df))
fig, axes = plt.subplots(1, len(multi_targets_valid), figsize=(6 * len(multi_targets_valid), 8))
axes = np.atleast_1d(axes)
for ax, tgt in zip(axes, multi_targets_valid):
    top_features = multi_corr_df[tgt].abs().sort_values(ascending=False).head(top_n).index
    vals = multi_corr_df.loc[top_features, tgt].values
    ax.barh(range(top_n), vals, color=['#2196F3' if v >= 0 else '#F44336' for v in vals])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Correlação de Pearson')
    ax.set_title(tgt, fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
fig.suptitle(f'Top {top_n} Features por Target — Pearson exploratório', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA — amostra temporal
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

tags_with_data = [c for c in df_merged.columns if df_merged[c].notna().sum() > 0][:4]

for i, tag in enumerate(tags_with_data):
    sample = df_merged[tag].dropna()
    if len(sample) > 0:
        sample_7d = sample.loc[sample.index[:24*7]]   # primeiros 7 dias
        axes[i].plot(sample_7d.index, sample_7d.values, linewidth=0.8)
        axes[i].set_ylabel(tag, fontsize=10)
        axes[i].tick_params(axis='x', rotation=30)

axes[0].set_title('Amostra Temporal (7 dias) — Primeiras 4 Tags de Processo', fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Definição dos targets e features base
multi_targets_valid = [t for t in MULTI_TARGETS if t in df_merged.columns]
multi_targets_ausentes = [t for t in MULTI_TARGETS if t not in df_merged.columns]

print("=== Targets (sistemas distintos) ===")
for t in multi_targets_valid:
    print(f"  {t}: {df_merged[t].notna().sum():,} registros não-nulos | "
          f"{df_merged[t].first_valid_index()} a {df_merged[t].last_valid_index()}")
if multi_targets_ausentes:
    print(f"\nTags não encontradas: {multi_targets_ausentes}")

TARGET_DERIVED = [c for c in df_merged.columns if is_target_derived(c)]
FEATURES_BASE = [c for c in df_merged.columns
                 if c not in multi_targets_valid
                 and c not in TARGET_DERIVED
                 and c not in EXCLUDE_FEATURES]

print(f"\nTotalizadores excluídos (_TOT): {TARGET_DERIVED}")
print(f"Gás/ar excluídos: {EXCLUDE_FEATURES}")
print(f">>> Features base (processo + clima): {len(FEATURES_BASE)}")

# COMMAND ----------

# DBTITLE 1,Feature engineering temporal (sem vazamento)
# Médias móveis calculadas sobre a série DESLOCADA (shift(HORIZON))
# -> não contêm y(t).

df_prep = df_merged.copy()
assert df_prep.index.freq is not None, "Rode a célula do merge (asfreq('h')) antes desta."

# Calendário
df_prep['hour'] = df_prep.index.hour
df_prep['dayofweek'] = df_prep.index.dayofweek
df_prep['month'] = df_prep.index.month
df_prep['dayofyear'] = df_prep.index.dayofyear
df_prep['hour_sin'] = np.sin(2 * np.pi * df_prep['hour'] / 24)
df_prep['hour_cos'] = np.cos(2 * np.pi * df_prep['hour'] / 24)
df_prep['dow_sin'] = np.sin(2 * np.pi * df_prep['dayofweek'] / 7)
df_prep['dow_cos'] = np.cos(2 * np.pi * df_prep['dayofweek'] / 7)
df_prep['month_sin'] = np.sin(2 * np.pi * df_prep['month'] / 12)
df_prep['month_cos'] = np.cos(2 * np.pi * df_prep['month'] / 12)
CALENDAR_FEATURES = ['hour', 'dayofweek', 'month', 'dayofyear',
                     'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

# Autorregressivos — somente passado
for tgt in multi_targets_valid:
    s = df_prep[tgt]
    for lag in LAGS:
        df_prep[f'{tgt}_lag_{lag}h'] = s.shift(lag)
    s_passado = s.shift(HORIZON)
    for window in MA_WINDOWS:
        df_prep[f'{tgt}_ma_{window}h'] = s_passado.rolling(window=window, min_periods=1).mean()

# Verificação de vazamento: a MA de 3h na hora t NÃO pode depender de y(t)
for tgt in multi_targets_valid:
    esperado = df_prep[tgt].shift(HORIZON).rolling(3, min_periods=1).mean()
    assert np.allclose(df_prep[f'{tgt}_ma_3h'], esperado, equal_nan=True)
print("OK: médias móveis usam apenas valores até t - HORIZON.")

print(f"DataFrame com features temporais: {df_prep.shape}")

# COMMAND ----------

# DBTITLE 1,Preparação final: conjuntos de features e travas
df_model = df_prep.dropna(subset=multi_targets_valid, how='all').copy()

FEATURES_FINAL = [c for c in df_model.columns
                  if c not in multi_targets_valid
                  and not is_target_derived(c)
                  and c not in EXCLUDE_FEATURES]
FEATURES_AUTOREG = [c for c in FEATURES_FINAL if '_lag_' in c or '_ma_' in c]
FEATURES_EXOG = [c for c in FEATURES_FINAL if c not in FEATURES_AUTOREG]

# Travas contra vazamento
assert not any(is_target_derived(c) for c in FEATURES_FINAL), "_TOT entrou nas features!"
assert not set(EXCLUDE_FEATURES) & set(FEATURES_FINAL), "gás/ar entrou nas features!"
assert not set(multi_targets_valid) & set(FEATURES_FINAL), "target entrou nas features!"

print(f"Dados para modelagem: {df_model.shape}")
print(f"Período: {df_model.index.min()} a {df_model.index.max()}")
print(f"Features completas: {len(FEATURES_FINAL)} "
      f"(exógenas: {len(FEATURES_EXOG)} | autorregressivas: {len(FEATURES_AUTOREG)})")

# Alerta: variável EXÓGENA quase idêntica ao target no mesmo instante = suspeita de vazamento
print("\n--- Checagem de vazamento em exógenas (|r| > 0,98 com o target) ---")
suspeitas = False
for tgt in multi_targets_valid:
    r = df_model[FEATURES_EXOG].corrwith(df_model[tgt]).abs()
    alto = r[r > 0.98]
    if len(alto):
        suspeitas = True
        print(f"  {tgt}: {alto.round(4).to_dict()}")
if not suspeitas:
    print("  Nenhuma feature com |r| > 0,98.")

split_idx = int(len(df_model) * 0.8)
print(f"\nSplit temporal global: treino até {df_model.index[split_idx - 1]} | "
      f"teste a partir de {df_model.index[split_idx]}")

# COMMAND ----------

# DBTITLE 1,Salvar tabela preparada (silver_prepared)
df_save = df_model.reset_index()

(spark.createDataFrame(df_save)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable("workspace.previsao_vapor.silver_prepared"))

print(f"Tabela preparada salva: workspace.previsao_vapor.silver_prepared")
print(f"Shape: {df_save.shape}")
print(f"Período: {df_save['datetime'].min()} a {df_save['datetime'].max()}")
print("Linhas por ano:", df_save.groupby(df_save['datetime'].dt.year).size().to_dict())
display(df_save.head(5))

# COMMAND ----------

# DBTITLE 1,Feature importance — Pearson × Spearman × Gini × Permutation
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

plt.rcParams['figure.dpi'] = 100

def categorize_feature(name):
    if '_lag_' in name:
        return 'Lags (autorregressivo)'
    if '_ma_' in name:
        return 'Médias móveis'
    if name in ('hour', 'hour_sin', 'hour_cos'):
        return 'Sazonalidade horária'
    if name in ('dayofweek', 'dow_sin', 'dow_cos'):
        return 'Sazonalidade semanal'
    if name in ('month', 'month_sin', 'month_cos', 'dayofyear'):
        return 'Sazonalidade mensal/anual'
    if name in ('temperature_C', 'relative_humidity_pct', 'surface_pressure_hPa', 'pressure_msl_hPa'):
        return 'Clima'
    return 'Processo (pressão, temperatura, controle)'

# Recarregar do Delta se as células anteriores não foram executadas
try:
    _ = multi_targets_valid, df_model, FEATURES_FINAL, FEATURES_EXOG
except NameError:
    df_model = spark.table("workspace.previsao_vapor.silver_prepared").toPandas()
    df_model['datetime'] = pd.to_datetime(df_model['datetime'])
    df_model = df_model.set_index('datetime').sort_index()
    multi_targets_valid = [t for t in MULTI_TARGETS if t in df_model.columns]
    FEATURES_FINAL = [c for c in df_model.columns
                      if c not in multi_targets_valid
                      and not is_target_derived(c)
                      and c not in EXCLUDE_FEATURES]
    FEATURES_AUTOREG = [c for c in FEATURES_FINAL if '_lag_' in c or '_ma_' in c]
    FEATURES_EXOG = [c for c in FEATURES_FINAL if c not in FEATURES_AUTOREG]
    print(f"Dados carregados do Delta table: {df_model.shape}")

def avaliar_target(tgt, features, n_repeats=10):
    mask = df_model[tgt].notna()
    X_t = df_model.loc[mask, features]
    y_t = df_model.loc[mask, tgt]

    split = int(len(X_t) * 0.8)
    X_tr, X_te = X_t.iloc[:split], X_t.iloc[split:]
    y_tr, y_te = y_t.iloc[:split], y_t.iloc[split:]

    # Colunas 100% NaN no treino são descartadas pelo SimpleImputer -> remover antes
    valid_cols = X_tr.columns[X_tr.notna().any()].tolist()
    X_tr, X_te = X_tr[valid_cols], X_te[valid_cols]

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15,
                                     random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    perm = permutation_importance(pipe, X_te, y_te, scoring='r2', n_repeats=n_repeats,
                                  random_state=RANDOM_STATE, n_jobs=-1)

    imp = pd.DataFrame({
        'feature': valid_cols,
        'pearson': X_tr.corrwith(y_tr, method='pearson').reindex(valid_cols).values,
        'spearman': X_tr.corrwith(y_tr, method='spearman').reindex(valid_cols).values,
        'gini': pipe.named_steps['rf'].feature_importances_,
        'perm_mean': perm.importances_mean,
        'perm_std': perm.importances_std,
    })
    imp['rank_pearson'] = imp['pearson'].abs().rank(ascending=False)
    imp['rank_spearman'] = imp['spearman'].abs().rank(ascending=False)
    imp['rank_gini'] = imp['gini'].rank(ascending=False)
    imp['rank_perm'] = imp['perm_mean'].rank(ascending=False)
    imp = imp.sort_values('perm_mean', ascending=False).reset_index(drop=True)

    rho = lambda a, b: spearmanr(imp[a], imp[b], nan_policy='omit')[0]
    metr = {
        'n_treino': len(X_tr), 'n_teste': len(X_te),
        'inicio_teste': X_te.index.min(),
        'R2_teste': r2_score(y_te, y_pred),
        'MAE_teste': mean_absolute_error(y_te, y_pred),
        'rho_pearson_x_perm': rho('rank_pearson', 'rank_perm'),
        'rho_spearman_x_perm': rho('rank_spearman', 'rank_perm'),
        'rho_gini_x_perm': rho('rank_gini', 'rank_perm'),
        'rho_pearson_x_gini': rho('rank_pearson', 'rank_gini'),
    }
    return metr, imp

CENARIOS = {'completo': FEATURES_FINAL, 'exogeno': FEATURES_EXOG}
resultados, metricas = {}, []
for cen, feats in CENARIOS.items():
    for tgt in multi_targets_valid:
        m, imp = avaliar_target(tgt, feats)
        resultados[(cen, tgt)] = imp
        metricas.append({'cenario': cen, 'target': tgt, **m})
        print(f"[{cen:8s}] {tgt:15s} R²={m['R2_teste']:.3f}  MAE={m['MAE_teste']:.2f}  "
              f"| rho(Pearson×Perm)={m['rho_pearson_x_perm']:.2f}")

df_metricas = pd.DataFrame(metricas)
print("\n=== Métricas e concordância entre rankings ===")
display(df_metricas.round(3))

for tgt in multi_targets_valid:
    print(f"\n=== {tgt} — cenário completo (top 20 por permutation importance) ===")
    display(resultados[('completo', tgt)].head(20).round(4))

# COMMAND ----------

# DBTITLE 1,Gráfico comparativo — 4 métricas normalizadas por target
def plot_comparativo(cenario, top_n=15):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    for ax, tgt in zip(axes.flatten(), multi_targets_valid):
        imp = resultados[(cenario, tgt)].head(top_n).iloc[::-1]
        norm = lambda s: s / s.abs().max() if s.abs().max() > 0 else s
        series = {
            '|Pearson|': norm(imp['pearson'].abs()),
            '|Spearman|': norm(imp['spearman'].abs()),
            'Gini (treino)': norm(imp['gini']),
            'Permutation (teste)': norm(imp['perm_mean'].clip(lower=0)),
        }
        y = np.arange(len(imp))
        h = 0.2
        for k, (lbl, vals) in enumerate(series.items()):
            ax.barh(y + (k - 1.5) * h, vals.values, height=h, label=lbl)
        ax.set_yticks(y)
        ax.set_yticklabels(imp['feature'], fontsize=8)
        ax.set_xlabel('Valor normalizado (máx = 1)')
        m = df_metricas.query('cenario == @cenario and target == @tgt').iloc[0]
        ax.set_title(f"{tgt} | R²={m['R2_teste']:.3f} | rho Pearson×Perm={m['rho_pearson_x_perm']:.2f}",
                     fontsize=10)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    fig.suptitle(f'Pearson × Spearman × Gini × Permutation — cenário {cenario} (top {top_n} por permutation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_comparativo('completo')
plot_comparativo('exogeno')

# COMMAND ----------

# DBTITLE 1,Comparação Pearson × Random Forest
TOP_K = 15

# --- 1. Sobreposição dos Top-K ---
linhas = []
for cen in CENARIOS:
    for tgt in multi_targets_valid:
        imp = resultados[(cen, tgt)]
        k = min(TOP_K, len(imp))
        top = {m: set(imp.nsmallest(k, f'rank_{m}')['feature'])
               for m in ['pearson', 'spearman', 'gini', 'perm']}
        linhas.append({
            'cenario': cen, 'target': tgt, 'K': k,
            'Pearson∩Gini': len(top['pearson'] & top['gini']),
            'Pearson∩Perm': len(top['pearson'] & top['perm']),
            'Spearman∩Perm': len(top['spearman'] & top['perm']),
            'Gini∩Perm': len(top['gini'] & top['perm']),
        })
df_overlap = pd.DataFrame(linhas)
print(f"=== Quantas features do Top-{TOP_K} coincidem entre os métodos ===")
display(df_overlap)

# --- 2. Top-K lado a lado ---
def top_lista(imp, col_name, k, absoluto=False):
    s = imp.set_index('feature')[col_name]
    s = s.reindex(s.abs().sort_values(ascending=False).index) if absoluto else s.sort_values(ascending=False)
    return [f"{f} ({v:+.3f})" if absoluto else f"{f} ({v:.3f})" for f, v in s.head(k).items()]

for cen in CENARIOS:
    for tgt in multi_targets_valid:
        imp = resultados[(cen, tgt)]
        k = min(TOP_K, len(imp))
        lado = pd.DataFrame({
            'Pearson r': top_lista(imp, 'pearson', k, absoluto=True),
            'Spearman ρ': top_lista(imp, 'spearman', k, absoluto=True),
            'RF Gini (treino)': top_lista(imp, 'gini', k),
            'RF Permutation (teste)': top_lista(imp, 'perm_mean', k),
        }, index=range(1, k + 1))
        print(f"\n=== {tgt} — cenário {cen}: Top-{k} por método ===")
        display(lado)

# --- 3. Dispersão |Pearson| × importância ---
cat_cores = {
    'Lags (autorregressivo)': '#1f77b4', 'Médias móveis': '#17becf',
    'Sazonalidade horária': '#2ca02c', 'Sazonalidade semanal': '#98df8a',
    'Sazonalidade mensal/anual': '#bcbd22', 'Clima': '#ff7f0e',
    'Processo (pressão, temperatura, controle)': '#d62728',
}
for cen in CENARIOS:
    fig, axes = plt.subplots(2, len(multi_targets_valid), figsize=(6 * len(multi_targets_valid), 10))
    axes = np.array(axes).reshape(2, -1)
    for j, tgt in enumerate(multi_targets_valid):
        imp = resultados[(cen, tgt)].copy()
        imp['categoria'] = imp['feature'].apply(categorize_feature)
        imp['abs_pearson'] = imp['pearson'].abs()
        for i, (col_name, lbl) in enumerate([('gini', 'RF Gini (treino)'), ('perm_mean', 'RF Permutation (teste)')]):
            ax = axes[i, j]
            for cat, g in imp.groupby('categoria'):
                ax.scatter(g['abs_pearson'], g[col_name], s=30, alpha=0.8, label=cat,
                           color=cat_cores.get(cat, 'grey'))
            for _, r in imp.nlargest(5, col_name).iterrows():
                ax.annotate(r['feature'], (r['abs_pearson'], r[col_name]), fontsize=7, alpha=0.8)
            ax.set_xlabel('|Pearson r| (treino)')
            ax.set_ylabel(lbl)
            ax.set_title(f'{tgt}', fontsize=10)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'|Pearson| × importância do Random Forest — cenário {cen}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- 4. Divergências ---
print("\n=== Divergências (cenário completo) ===")
print("A: alta correlação linear, pouca utilidade no RF -> em geral redundância/colinearidade")
print("B: baixa correlação linear, muita utilidade no RF -> em geral relação não linear/interação/limiar")
for tgt in multi_targets_valid:
    imp = resultados[('completo', tgt)]
    a = imp[(imp['rank_pearson'] <= 10) & (imp['rank_perm'] > 20)]
    b = imp[(imp['rank_perm'] <= 10) & (imp['rank_pearson'] > 20)]
    print(f"\n{tgt}")
    print(f"  A: {a['feature'].tolist() or '—'}")
    print(f"  B: {b['feature'].tolist() or '—'}")

# COMMAND ----------

# DBTITLE 1,Ranking consolidado + importância por categoria
def consolidar(cenario):
    frames = []
    for tgt in multi_targets_valid:
        imp = resultados[(cenario, tgt)].set_index('feature')
        frames.append(imp[['gini', 'perm_mean']].add_suffix(f'|{tgt}'))
        frames.append(imp[['pearson']].abs().rename(columns={'pearson': f'abs_pearson|{tgt}'}))
    comb = pd.concat(frames, axis=1)
    gini_cols = [c for c in comb if c.startswith('gini|')]
    perm_cols = [c for c in comb if c.startswith('perm_mean|')]
    pear_cols = [c for c in comb if c.startswith('abs_pearson|')]
    out = pd.DataFrame({
        'gini_media': comb[gini_cols].fillna(0).mean(axis=1),
        'perm_media': comb[perm_cols].fillna(0).mean(axis=1),
        'abs_pearson_media': comb[pear_cols].mean(axis=1),
    })
    out['categoria'] = [categorize_feature(f) for f in out.index]
    return out.sort_values('perm_media', ascending=False)

for cen in CENARIOS:
    cons = consolidar(cen)
    print(f"\n{'=' * 70}\nTOP 15 CONSOLIDADO — cenário {cen} (ordenado por permutation)\n{'=' * 70}")
    display(cons.head(15).round(4))
    cat = cons.groupby('categoria')[['perm_media', 'gini_media']].sum().sort_values('perm_media', ascending=False)
    cat['n_features'] = cons.groupby('categoria').size()
    print(f"\nImportância por categoria — cenário {cen}")
    display(cat.round(4))