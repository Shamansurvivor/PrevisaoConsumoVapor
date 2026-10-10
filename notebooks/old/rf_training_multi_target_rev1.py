# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Treinamento Random Forest — Previsão de Consumo de Vapor (Multi-Target) — rev1
# MAGIC
# MAGIC Treina um Random Forest independente para cada um dos 4 sistemas de consumo de vapor
# MAGIC (`11FI208-02`, `11FI502-30.PV`, `21FI550-30`, `21FI551-10`) e para o consumo total, usando
# MAGIC `workspace.previsao_vapor.silver_prepared` (gerada por `random_forest_prep_rev2`). Avalia com RMSE, MAE, R² e MAPE,
# MAGIC faz previsão de 30 dias (RF horário recursivo, RF diário, Prophet, SARIMA) e combina em ensemble.
# MAGIC
# MAGIC **rev1 — o que mudou em relação ao rev0**
# MAGIC - Período configurável (`DATA_INICIO = 2023-01-01`, `DATA_FIM = None` = último dado) + trava contra `silver_prepared` desatualizada.
# MAGIC - **Vazamento corrigido:** totalizadores `_TOT`/`_TOT_DAY` e gás/ar de combustão (`11FC550-72.PV`, `11FC208-106.PV`) não entram mais como feature.
# MAGIC - `%pip install` movido para o início (não apaga mais o estado no meio do notebook).
# MAGIC - Prophet e SARIMA leem a mesma tabela Delta (antes liam um CSV antigo de outro período).
# MAGIC - Feriados gerados pela biblioteca `holidays` para todos os anos do histórico (antes só 2026).
# MAGIC - Ensemble e comparações usam as previsões calculadas no próprio notebook (antes eram valores digitados de uma execução antiga, com data fixa 2026-07-01).
# MAGIC - Célula que gera `pred_df` / `real_total_inst` reconstruída (predições dos 4 modelos no período de teste).
# MAGIC
# MAGIC **Ordem:** rodar `random_forest_prep_rev2` antes deste notebook.

# COMMAND ----------

# MAGIC %pip install prophet seaborn holidays --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import holidays
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("Bibliotecas importadas com sucesso.")

# COMMAND ----------

# DBTITLE 1,Configuração central
# --- Período de dados ---
DATA_INICIO = pd.Timestamp("2023-01-01 00:00:00")
DATA_FIM = None          # None = até o último dado coletado

# --- Targets e exclusões (mesmo critério do random_forest_prep_rev2) ---
MULTI_TARGETS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']
EXCLUDE_FEATURES = ['11FC550-72.PV', '11FC208-106.PV']   # gás/ar de combustão = proxy do target

def is_target_derived(c):
    """Totalizadores (_TOT, _TOT_DAY) nunca entram como feature."""
    return '_TOT' in c

# --- Tabelas / caminhos ---
SILVER_TABLE = "workspace.previsao_vapor.silver_prepared"
MODELS_DIR = "/Volumes/workspace/previsao_vapor/models"
RESULTS_TABLE = "workspace.previsao_vapor.gold_results_summary"
ENSEMBLE_TABLE = "workspace.previsao_vapor.gold_ensemble_forecast_30d"
MLFLOW_EXPERIMENT = "/Previsao de consumo de vapor/rf_hyperopt"

# --- Previsão ---
FORECAST_DAYS = 30
TEST_FRACTION = 0.2
CLIMATE_COLS = ['temperature_C', 'relative_humidity_pct', 'surface_pressure_hPa', 'pressure_msl_hPa']
RANDOM_STATE = 42

print(f"Período: {DATA_INICIO} -> {'último dado coletado' if DATA_FIM is None else DATA_FIM}")

# COMMAND ----------

# DBTITLE 1,Carregar dados preparados
df = spark.table(SILVER_TABLE).toPandas()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()
df = df.loc[DATA_INICIO:] if DATA_FIM is None else df.loc[DATA_INICIO:DATA_FIM]

assert df.index.min() < DATA_INICIO + pd.Timedelta(days=7), (
    f"{SILVER_TABLE} começa em {df.index.min()} — rode random_forest_prep_rev2 antes deste notebook."
)

print(f"Dados carregados: {df.shape}")
print(f"Período: {df.index.min()} a {df.index.max()}")
print(f"Colunas: {len(df.columns)}")
print("Linhas por ano:", df.groupby(df.index.year).size().to_dict())
display(df.head(3))

# COMMAND ----------

# DBTITLE 1,Definir targets e features
multi_targets_valid = [t for t in MULTI_TARGETS if t in df.columns]

FEATURES = [c for c in df.columns
            if c not in multi_targets_valid
            and not is_target_derived(c)
            and c not in EXCLUDE_FEATURES]

# Travas contra vazamento
assert not any(is_target_derived(c) for c in FEATURES), "_TOT entrou nas features!"
assert not set(EXCLUDE_FEATURES) & set(FEATURES), "gás/ar entrou nas features!"
assert not set(multi_targets_valid) & set(FEATURES), "target entrou nas features!"

print(f"Targets: {multi_targets_valid}")
print(f"Features ({len(FEATURES)})")
for tgt in multi_targets_valid:
    print(f"  {tgt}: {df[tgt].notna().sum():,} não-nulos | "
          f"{df[tgt].first_valid_index()} a {df[tgt].last_valid_index()}")

# COMMAND ----------

# DBTITLE 1,Funções auxiliares
def mape(y_true, y_pred):
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    return float(np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100)

def split_target(tgt, features):
    """Filtra linhas com target, split temporal 80/20, remove colunas 100% NaN no treino."""
    mask = df[tgt].notna()
    X = df.loc[mask, features]
    y = df.loc[mask, tgt]
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    cols = X_train.columns[X_train.notna().any()].tolist()
    return X_train[cols], X_test[cols], y_train, y_test, cols

def make_pipeline(**rf_params):
    params = dict(n_estimators=100, max_depth=20, min_samples_leaf=5,
                  random_state=RANDOM_STATE, n_jobs=-1)
    params.update(rf_params)
    return Pipeline([('imputer', SimpleImputer(strategy='mean')),
                     ('rf', RandomForestRegressor(**params))])

def plot_pred_e_importancia(tgt, y_test, y_pred_test, feat_imp, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Predito')
    axes[0].set_title(f'{tgt} — Predito vs Real (teste)')
    top_n = min(top_n, len(feat_imp))
    top_imp = feat_imp.head(top_n)
    axes[1].barh(range(top_n), top_imp['importance'].values, edgecolor='white')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_imp['feature'].values, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importância')
    axes[1].set_title(f'{tgt} — Top {top_n} Features')
    plt.tight_layout()
    plt.show()

def treinar_e_avaliar(tgt, features):
    X_train, X_test, y_train, y_test, cols = split_target(tgt, features)
    print(f"  Treino: {X_train.shape[0]:,} ({X_train.index.min().date()} a {X_train.index.max().date()}) | "
          f"Teste: {X_test.shape[0]:,} ({X_test.index.min().date()} a {X_test.index.max().date()}) | "
          f"Features: {len(cols)}")

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    res = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'r2_test': r2_score(y_test, y_pred_test),
        'mape_test': mape(y_test, y_pred_test),
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
    }
    for k in ['rmse_train', 'rmse_test', 'mae_test']:
        print(f"  {k.upper():11s}: {res[k]:.4f}")
    print(f"  R2 teste   : {res['r2_test']:.4f}")
    print(f"  MAPE teste : {res['mape_test']:.2f}%")

    feat_imp = pd.DataFrame({'feature': cols,
                             'importance': pipeline.named_steps['rf'].feature_importances_}) \
                 .sort_values('importance', ascending=False)
    print("\n  Top 10 features:")
    for _, row in feat_imp.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    plot_pred_e_importancia(tgt, y_test, y_pred_test, feat_imp)
    return pipeline, res

def get_model_cols(model):
    """Colunas usadas no treino do pipeline."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    imputer = model.named_steps['imputer']
    return list(imputer.feature_names_in_) if hasattr(imputer, 'feature_names_in_') else FEATURES

# COMMAND ----------

# DBTITLE 1,Treinamento de um Random Forest para cada target
results = {}
models = {}

for tgt in multi_targets_valid:
    print(f"\n{'='*60}\nTREINANDO RF PARA: {tgt}\n{'='*60}")
    models[tgt], results[tgt] = treinar_e_avaliar(tgt, FEATURES)

print("\n" + "="*60 + "\nTREINAMENTO CONCLUÍDO PARA TODOS OS TARGETS\n" + "="*60)

# COMMAND ----------

# DBTITLE 1,Tabela comparativa de resultados
results_df = pd.DataFrame(results).T.astype(float).round(4)

print("=== Comparação de modelos por target ===")
display(results_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
results_df[['rmse_test', 'mae_test']].plot.bar(ax=axes[0], edgecolor='white')
axes[0].set_title('RMSE e MAE (teste) por Target')
axes[0].set_ylabel('Erro')
axes[0].tick_params(axis='x', rotation=30)

results_df['r2_test'].plot.bar(ax=axes[1], color='#4CAF50', edgecolor='white')
axes[1].set_title('R2 (teste) por Target')
axes[1].set_ylabel('R2')
axes[1].tick_params(axis='x', rotation=30)
axes[1].set_ylim(min(0, results_df['r2_test'].min() - 0.05), 1)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Otimização de hiperparâmetros (RandomizedSearchCV + TimeSeriesSplit)
# rev1.1: versão leve. A anterior (n_jobs=-1 na busca E no RF, grade com árvores sem
# limite de profundidade e max_features=0.8) podia levar horas ou travar no serverless.
#  - paralelismo só na busca (RF interno com n_jobs=1, sem disputa de núcleos)
#  - grade reduzida, sem max_depth=None / min_samples_leaf=1
#  - busca numa amostra do treino (1 a cada HYPEROPT_STEP horas); o melhor conjunto
#    é re-treinado no treino completo
#  - progresso e tempo impressos por target; RUN_HYPEROPT=False pula a célula
import time

RUN_HYPEROPT = True
HYPEROPT_N_ITER = 6
HYPEROPT_STEP = 3          # usa 1 a cada 3 horas na busca (só na busca)
LOG_MODEL_MLFLOW = False   # True = também grava o modelo no MLflow (mais lento)

param_distributions = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 15, 20, 30],
    'rf__min_samples_leaf': [3, 5, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 0.3, 0.5],
}

results_before = {t: dict(results[t]) for t in multi_targets_valid}

if not RUN_HYPEROPT:
    print("Otimização desativada (RUN_HYPEROPT = False) — mantidos os modelos default.")
else:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    for tgt in multi_targets_valid:
        t0 = time.time()
        print(f"\n{'='*60}\nOTIMIZANDO RF PARA: {tgt}\n{'='*60}")
        X_train, X_test, y_train, y_test, cols = split_target(tgt, FEATURES)
        X_s, y_s = X_train.iloc[::HYPEROPT_STEP], y_train.iloc[::HYPEROPT_STEP]
        print(f"  Busca em {len(X_s):,} de {len(X_train):,} linhas de treino | "
              f"{HYPEROPT_N_ITER} combinações x 3 folds")

        pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('rf', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))])
        search = RandomizedSearchCV(pipeline, param_distributions, n_iter=HYPEROPT_N_ITER,
                                    cv=TimeSeriesSplit(n_splits=3),
                                    scoring='neg_root_mean_squared_error', n_jobs=-1,
                                    random_state=RANDOM_STATE, verbose=1)
        search.fit(X_s, y_s)
        print(f"  Busca concluída em {(time.time() - t0)/60:.1f} min")

        # Re-treina o melhor conjunto no treino completo
        best_params = {k.replace('rf__', ''): v for k, v in search.best_params_.items()}
        best_model = make_pipeline(**best_params)
        best_model.fit(X_train, y_train)

        y_pred_test = best_model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        mape_test = mape(y_test, y_pred_test)

        safe_name = tgt.replace('.', '_').replace('-', '_')
        try:
            with mlflow.start_run(run_name=f"rf_opt_{safe_name}"):
                mlflow.log_params(best_params)
                mlflow.log_params({'data_inicio': str(X_train.index.min()),
                                   'data_fim': str(X_test.index.max()), 'n_features': len(cols)})
                mlflow.log_metrics({'rmse_test': rmse_test, 'mae_test': mae_test,
                                    'r2_test': r2_test, 'mape_test': mape_test})
                if LOG_MODEL_MLFLOW:
                    mlflow.sklearn.log_model(best_model, f"rf_{safe_name}",
                                             input_example=X_train.head(5))
        except Exception as e:
            print(f"  (MLflow não registrou: {e})")

        # Só substitui o modelo default se o otimizado for melhor no teste
        if rmse_test < results_before[tgt]['rmse_test']:
            results[tgt] = {
                'rmse_train': np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
                'rmse_test': rmse_test, 'mae_test': mae_test, 'r2_test': r2_test, 'mape_test': mape_test,
                'n_train': X_train.shape[0], 'n_test': X_test.shape[0],
                'best_params': str(best_params),
            }
            models[tgt] = best_model
            print("  -> modelo otimizado adotado")
        else:
            print("  -> modelo default mantido (otimizado não melhorou o RMSE de teste)")

        print(f"  Melhores params: {best_params}")
        print(f"  RMSE: {results_before[tgt]['rmse_test']:.4f} -> {rmse_test:.4f}")
        print(f"  R2:   {results_before[tgt]['r2_test']:.4f} -> {r2_test:.4f}")
        print(f"  MAPE: {results_before[tgt]['mape_test']:.2f}% -> {mape_test:.2f}%")
        print(f"  Tempo total do target: {(time.time() - t0)/60:.1f} min")

comp_opt = pd.DataFrame({
    'RMSE default': {t: results_before[t]['rmse_test'] for t in multi_targets_valid},
    'RMSE final': {t: results[t]['rmse_test'] for t in multi_targets_valid},
    'R2 default': {t: results_before[t]['r2_test'] for t in multi_targets_valid},
    'R2 final': {t: results[t]['r2_test'] for t in multi_targets_valid},
}).round(4)
print("\n=== Comparação: Default vs Final ===")
display(comp_opt)

# COMMAND ----------

# DBTITLE 1,Calcular consumo total de vapor (instantâneo + diário)
# Consumo instantâneo = soma das 4 vazões (FI)
# Consumo diário = soma dos 4 totalizadores (_TOT); tags sem _TOT -> resample diário do instantâneo
FI_TAGS = multi_targets_valid

TOT_MAPPING = {}
for fi in FI_TAGS:
    fi_base = fi.replace('.PV', '')
    candidates = [f'{fi}_TOT_DAY', f'{fi}_TOT', f'{fi_base}_TOT_DAY', f'{fi_base}_TOT']
    TOT_MAPPING[fi] = next((c for c in candidates if c in df.columns), None)

print("=== Mapeamento FI → _TOT ===")
for fi, tot in TOT_MAPPING.items():
    print(f"  {fi} → {tot} ({df[tot].notna().sum():,} registros)" if tot
          else f"  {fi} → ausente (calculado pelo instantâneo)")

df_tot_daily = pd.DataFrame(index=df.index)
for fi in FI_TAGS:
    tot_col = TOT_MAPPING.get(fi)
    if tot_col is not None:
        df_tot_daily[fi] = df[tot_col]
    else:
        daily_sum = df[fi].resample('D').sum(min_count=1)
        df_tot_daily[fi] = daily_sum.reindex(df.index, method='ffill')

df['TOTAL_INST'] = df[FI_TAGS].sum(axis=1, min_count=1)
df['TOTAL_DAY'] = df_tot_daily.sum(axis=1, min_count=1)

print(f"\n=== Consumo Total de Vapor ===")
for c in ['TOTAL_INST', 'TOTAL_DAY']:
    print(f"  {c}: {df[c].notna().sum():,} registros | min={df[c].min():.2f}, "
          f"max={df[c].max():.2f}, mean={df[c].mean():.2f}")

total_targets = ['TOTAL_INST', 'TOTAL_DAY']
total_features = [c for c in df.columns
                  if c not in multi_targets_valid
                  and c not in total_targets
                  and not is_target_derived(c)
                  and c not in EXCLUDE_FEATURES]
assert not any(is_target_derived(c) for c in total_features)

print(f"\n>>> Targets totais: {total_targets}")
print(f">>> Features ({len(total_features)})")

# COMMAND ----------

# DBTITLE 1,Treinar RF para consumo total
for tgt in total_targets:
    print(f"\n{'='*60}\nTREINANDO RF PARA: {tgt}\n{'='*60}")
    models[tgt], results[tgt] = treinar_e_avaliar(tgt, total_features)

print("\n" + "="*60 + "\nTREINAMENTO CONCLUÍDO — INDIVIDUAL + TOTAL\n" + "="*60)

# COMMAND ----------

# DBTITLE 1,Salvar modelos (UC Volume) e resultados (Delta)
os.makedirs(MODELS_DIR, exist_ok=True)
for tgt, model in models.items():
    safe_name = tgt.replace('.', '_').replace('-', '_')
    model_path = f"{MODELS_DIR}/rf_{safe_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo salvo: {model_path}")

results_df = pd.DataFrame(results).T
num_cols = ['rmse_train', 'rmse_test', 'mae_test', 'r2_test', 'mape_test', 'n_train', 'n_test']
results_save = results_df[num_cols].astype(float).round(4)
results_save['best_params'] = results_df.get('best_params', pd.Series(index=results_df.index, dtype=str)) \
                                        .fillna('default').astype(str)
results_save['data_inicio'] = str(df.index.min())
results_save['data_fim'] = str(df.index.max())
results_save['run_ts'] = pd.Timestamp.now().isoformat()

(spark.createDataFrame(results_save.reset_index().rename(columns={'index': 'modelo'}))
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RESULTS_TABLE))
print(f"\nResultados salvos: {RESULTS_TABLE}")
display(results_save)

# COMMAND ----------

# DBTITLE 1,Predições dos 4 modelos no período de teste (pred_df)
# Reconstruída no rev1: o rev0 usava pred_df / real_total_inst sem célula que os criasse.
# Período de teste global = últimos TEST_FRACTION das horas; cada modelo prevê esse período
# e a soma das 4 predições é comparada com o consumo total real.
test_start = df.index[int(len(df) * (1 - TEST_FRACTION))]
df_test = df.loc[test_start:]

pred_df = pd.DataFrame(index=df_test.index)
for tgt in multi_targets_valid:
    cols = get_model_cols(models[tgt])
    pred = pd.Series(models[tgt].predict(df_test[cols]), index=df_test.index)
    pred_df[f'{tgt}_pred'] = pred.where(df_test[tgt].notna())   # só onde há medição real

pred_cols = [f'{t}_pred' for t in multi_targets_valid]
pred_df['TOTAL_INST_pred_soma'] = pred_df[pred_cols].sum(axis=1, min_count=1)
real_total_inst = df_test['TOTAL_INST']

mask = pred_df['TOTAL_INST_pred_soma'].notna() & real_total_inst.notna()
print(f"Período de teste: {test_start} a {df_test.index.max()} ({mask.sum():,} horas válidas)")
print(f"  RMSE (soma das 4 predições): {np.sqrt(mean_squared_error(real_total_inst[mask], pred_df.loc[mask, 'TOTAL_INST_pred_soma'])):.4f}")
print(f"  R2:   {r2_score(real_total_inst[mask], pred_df.loc[mask, 'TOTAL_INST_pred_soma']):.4f}")
print(f"  MAPE: {mape(real_total_inst[mask], pred_df.loc[mask, 'TOTAL_INST_pred_soma']):.2f}%")

# COMMAND ----------

# DBTITLE 1,Previsão do consumo diário por agregação das predições horárias
pred_hourly = pred_df['TOTAL_INST_pred_soma'].dropna()
real_hourly = real_total_inst.dropna()

common_idx = pred_hourly.index.intersection(real_hourly.index)
pred_hourly = pred_hourly.loc[common_idx]
real_hourly = real_hourly.loc[common_idx]

pred_daily = pred_hourly.resample('D').sum()
real_daily = real_hourly.resample('D').sum()

# Apenas dias com pelo menos 12 horas de dados (evitar dias parciais)
hours_per_day = real_hourly.resample('D').count()
valid_days = hours_per_day[hours_per_day >= 12].index
pred_daily = pred_daily.loc[valid_days]
real_daily = real_daily.loc[valid_days]

rmse_daily = np.sqrt(mean_squared_error(real_daily, pred_daily))
mae_daily = mean_absolute_error(real_daily, pred_daily)
r2_daily = r2_score(real_daily, pred_daily)
mape_daily = mape(real_daily, pred_daily)

print("=== PREVISÃO DO CONSUMO DIÁRIO (agregação horária) ===")
print(f"  Dias avaliados: {len(valid_days)}")
print(f"  RMSE: {rmse_daily:.2f} | MAE: {mae_daily:.2f} | R2: {r2_daily:.4f} | MAPE: {mape_daily:.2f}%")
print(f"\n  R2 modelo direto TOTAL_DAY: {results.get('TOTAL_DAY', {}).get('r2_test', float('nan')):.4f}")
print(f"  R2 agregação horária:       {r2_daily:.4f}")

comp_daily = pd.DataFrame({
    'real': real_daily, 'predito': pred_daily, 'erro': real_daily - pred_daily,
    'erro_pct': ((real_daily - pred_daily) / real_daily * 100).round(1),
})
print(f"\n=== Amostra: primeiros 10 dias ===")
display(comp_daily.head(10).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(real_daily, pred_daily, alpha=0.6, s=30)
axes[0].plot([real_daily.min(), real_daily.max()], [real_daily.min(), real_daily.max()], 'r--', linewidth=1)
axes[0].set_xlabel('Real (consumo diário)')
axes[0].set_ylabel('Predito')
axes[0].set_title(f'Consumo Diário — Real vs Predito (R2={r2_daily:.3f})')
axes[1].plot(real_daily.index, real_daily.values, label='Real', color='black', linewidth=1.5)
axes[1].plot(pred_daily.index, pred_daily.values, label='Predito (agregado)', color='#2196F3', alpha=0.8)
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Consumo diário de vapor')
axes[1].set_title('Consumo Diário — Real vs Predito')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Previsão multi-passo (recursiva) — RF horário
# Prevê H horas à frente: a predição de cada passo alimenta os lags do passo seguinte.
# A soma das 4 predições individuais dá o consumo total previsto.
H = FORECAST_DAYS * 24
H_EVAL = 168   # walk-forward (7 dias)

TEMPORAL_COLS = ['hour', 'dayofweek', 'month', 'dayofyear',
                 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

def build_feature_row(ts, model_cols, last_row, hist_dict, targets):
    row = {}
    for c in model_cols:
        if c in TEMPORAL_COLS:
            row[c] = {'hour': ts.hour, 'dayofweek': ts.dayofweek, 'month': ts.month,
                      'dayofyear': ts.dayofyear,
                      'hour_sin': np.sin(2 * np.pi * ts.hour / 24), 'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
                      'dow_sin': np.sin(2 * np.pi * ts.dayofweek / 7), 'dow_cos': np.cos(2 * np.pi * ts.dayofweek / 7),
                      'month_sin': np.sin(2 * np.pi * ts.month / 12), 'month_cos': np.cos(2 * np.pi * ts.month / 12)}[c]
        elif any(c.startswith(f'{t}_lag_') for t in targets):
            t = next(t for t in targets if c.startswith(f'{t}_lag_'))
            lag_n = int(c.split('_lag_')[1].replace('h', ''))
            h = hist_dict[t]
            row[c] = h[-lag_n] if len(h) >= lag_n else h[0]
        elif any(c.startswith(f'{t}_ma_') for t in targets):
            t = next(t for t in targets if c.startswith(f'{t}_ma_'))
            ma_n = int(c.split('_ma_')[1].replace('h', ''))
            h = hist_dict[t]
            recent = h[-ma_n:] if len(h) >= ma_n else h
            row[c] = np.mean(recent) if recent else 0
        else:
            row[c] = last_row[c] if c in last_row.index else np.nan   # exógenas: último valor conhecido
    return row

def recursive_forecast(start_ts, H, df, models, targets):
    hist_dict = {}
    for tgt in targets:
        vals = df[tgt].dropna()
        hist_dict[tgt] = vals.loc[vals.index < start_ts].tail(24).values.tolist()
        if len(hist_dict[tgt]) < 24:
            hist_dict[tgt] = hist_dict[tgt] + [0] * (24 - len(hist_dict[tgt]))

    last_row = df.loc[df.index < start_ts].iloc[-1].copy()
    future_ts = pd.date_range(start=start_ts, periods=H, freq='h')
    model_cols = {tgt: get_model_cols(models[tgt]) for tgt in targets}
    forecasts = {tgt: [] for tgt in targets}

    for ts in future_ts:
        preds_step = {}
        for tgt in targets:
            fr = build_feature_row(ts, model_cols[tgt], last_row, hist_dict, targets)
            preds_step[tgt] = models[tgt].predict(pd.DataFrame([fr], columns=model_cols[tgt]))[0]
            forecasts[tgt].append(preds_step[tgt])
        for tgt in targets:
            hist_dict[tgt].append(preds_step[tgt])
            hist_dict[tgt] = hist_dict[tgt][-48:]

    fdf = pd.DataFrame(forecasts, index=future_ts)
    fdf['TOTAL_INST_pred'] = fdf[targets].sum(axis=1)
    return fdf

# Predição de 1 linha por vez: n_jobs=1 evita o custo de abrir threads a cada passo
for _m in models.values():
    _m.named_steps['rf'].set_params(n_jobs=1)

# --- 1. Previsão a partir do último dado (começa na próxima meia-noite -> dias completos) ---
last_ts = df.index.max()
start_fc = last_ts.normalize() + pd.Timedelta(days=1)
print(f"=== PREVISÃO MULTI-PASSO ({H}h à frente) ===")
print(f"Último dado: {last_ts} | início da previsão: {start_fc}")

forecast = recursive_forecast(start_fc, H, df, models, multi_targets_valid)
forecast_daily = forecast['TOTAL_INST_pred'].resample('D').sum()

print(f"\nPrevisão horária (primeiras 6h):")
display(forecast.head(6).round(4))
print(f"\nConsumo total previsto por dia ({FORECAST_DAYS} dias):")
for date, val in forecast_daily.items():
    print(f"  {date.date()}: {val:.2f}")
print(f"\nTotal {FORECAST_DAYS} dias: {forecast_daily.sum():.2f}")
print(f"Média diária: {forecast_daily.mean():.2f}")

# --- 2. Avaliação walk-forward no período de teste ---
test_df = df.loc[test_start:]
sample_step = max(200, len(test_df) // 8)
start_points = list(range(24, len(test_df) - H_EVAL, sample_step))
errors_by_h = {h: [] for h in range(1, H_EVAL + 1)}

for start in start_points:
    start_ts = test_df.index[start]
    try:
        fc = recursive_forecast(start_ts, H_EVAL, df, models, multi_targets_valid)
    except Exception as e:
        print(f"  falha em {start_ts}: {e}")
        continue
    for h in range(H_EVAL):
        actual_ts = fc.index[h]
        if actual_ts in test_df.index:
            actual = test_df.loc[actual_ts, 'TOTAL_INST']
            pred = fc.iloc[h]['TOTAL_INST_pred']
            if pd.notna(actual) and pd.notna(pred):
                errors_by_h[h + 1].append(abs(actual - pred))

mae_by_h = {h: np.mean(errs) for h, errs in errors_by_h.items() if errs}

print(f"\n=== AVALIAÇÃO WALK-FORWARD (teste, {H_EVAL}h) ===")
print(f"Pontos de início avaliados: {len(start_points)}")
for h in sorted(mae_by_h):
    if h % 24 == 0 or h == 1:
        print(f"  h={h:3d}: MAE={mae_by_h[h]:.4f} (n={len(errors_by_h[h])})")

# --- 3. Gráficos ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
axes[0].bar(range(len(forecast_daily)), forecast_daily.values, color='#4CAF50', edgecolor='white')
axes[0].set_xticks(range(len(forecast_daily)))
axes[0].set_xticklabels([d.strftime('%m-%d') for d in forecast_daily.index], rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diário')
axes[0].set_title(f'Previsão do Consumo Diário de Vapor ({FORECAST_DAYS} dias a partir de {start_fc.date()})')
axes[0].axhline(y=forecast_daily.mean(), color='red', linestyle='--', alpha=0.5,
                label=f'Média: {forecast_daily.mean():.1f}')
axes[0].legend()

n_plot = min(168, H)
axes[1].plot(forecast.index[:n_plot], forecast['TOTAL_INST_pred'][:n_plot], color='black', linewidth=1.5)
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Consumo instantâneo')
axes[1].set_title('Previsão Horária do Total (primeiros 7 dias)')
axes[1].tick_params(axis='x', rotation=30)

if mae_by_h:
    horizons = sorted(mae_by_h)
    maes = [mae_by_h[h] for h in horizons]
    axes[2].plot(horizons, maes, color='#FF9800', linewidth=1.5)
    axes[2].fill_between(horizons, 0, maes, alpha=0.2, color='#FF9800')
    for d in range(1, H_EVAL // 24 + 1):
        axes[2].axvline(x=d * 24, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_xticks([1] + [d * 24 for d in range(1, H_EVAL // 24 + 1)])
axes[2].set_xlabel('Horizonte (horas à frente)')
axes[2].set_ylabel('MAE')
axes[2].set_title(f'Erro Absoluto por Horizonte (walk-forward {H_EVAL}h)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Série diária de referência (usada por RF diário, Prophet, SARIMA)
hourly_total = df['TOTAL_INST'].dropna()
hours_per_day_all = hourly_total.resample('D').count()
daily_total = hourly_total.resample('D').sum()
daily_total = daily_total.where(hours_per_day_all >= 12)   # dias com < 12 h viram NaN

daily_climate = pd.DataFrame({c: df[c].resample('D').mean() for c in CLIMATE_COLS if c in df.columns})
media_geral = daily_total.mean()

print(f"Dias no período: {len(daily_total)} | dias válidos (>= 12 h): {daily_total.notna().sum()}")
print(f"Período: {daily_total.first_valid_index().date()} a {daily_total.last_valid_index().date()}")
print(f"Consumo médio diário: {media_geral:.2f}")
print("Dias com clima ausente:", int(daily_climate.isna().any(axis=1).sum()))

# COMMAND ----------

# DBTITLE 1,Modelo diário com sazonalidade mensal — RF diário
daily = daily_total.to_frame('daily_total').join(daily_climate)
climate_cols = [c for c in CLIMATE_COLS if c in daily.columns]

daily['dayofweek'] = daily.index.dayofweek
daily['month'] = daily.index.month
daily['dayofyear'] = daily.index.dayofyear
daily['dow_sin'] = np.sin(2 * np.pi * daily['dayofweek'] / 7)
daily['dow_cos'] = np.cos(2 * np.pi * daily['dayofweek'] / 7)
daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
daily['dayofyear_sin'] = np.sin(2 * np.pi * daily['dayofyear'] / 365.25)
daily['dayofyear_cos'] = np.cos(2 * np.pi * daily['dayofyear'] / 365.25)

LAGS_D = [1, 2, 7, 14, 30]
MAS_D = [7, 14, 30]
for lag in LAGS_D:
    daily[f'lag_{lag}d'] = daily['daily_total'].shift(lag)
for ma in MAS_D:
    daily[f'ma_{ma}d'] = daily['daily_total'].shift(1).rolling(ma, min_periods=max(1, ma // 2)).mean()

# Sazonalidade histórica
monthly_avg = daily.groupby('month')['daily_total'].mean()
monthly_temp = daily.groupby('month')['temperature_C'].mean() if 'temperature_C' in daily.columns else None
print("=== Sazonalidade: consumo médio diário por mês (histórico) ===")
for m, v in monthly_avg.items():
    temp_str = f"  temp={monthly_temp[m]:.1f}C" if monthly_temp is not None else ""
    print(f"  Mês {m:2d}: {v:.2f}{temp_str}")
print(f"\n  Média geral: {media_geral:.2f}")
print(f"  Variação mensal: {monthly_avg.min():.2f} a {monthly_avg.max():.2f} "
      f"(amplitude: {monthly_avg.max() - monthly_avg.min():.2f})")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly_avg.index, monthly_avg.values, color='#2196F3', edgecolor='white')
ax.axhline(y=media_geral, color='red', linestyle='--', alpha=0.5, label=f'Média: {media_geral:.1f}')
ax.set_xlabel('Mês')
ax.set_ylabel('Consumo médio diário')
ax.set_title('Sazonalidade Mensal do Consumo de Vapor')
ax.set_xticks(monthly_avg.index)
ax.legend()
plt.tight_layout()
plt.show()

# Treino (clima ausente é imputado; só exige target e lags)
daily_features = [c for c in daily.columns if c != 'daily_total']
lag_ma_cols = [f'lag_{l}d' for l in LAGS_D] + [f'ma_{m}d' for m in MAS_D]
daily_model_data = daily.dropna(subset=['daily_total'] + lag_ma_cols)

split_d = int(len(daily_model_data) * (1 - TEST_FRACTION))
X_d_train = daily_model_data[daily_features].iloc[:split_d]
y_d_train = daily_model_data['daily_total'].iloc[:split_d]
X_d_test = daily_model_data[daily_features].iloc[split_d:]
y_d_test = daily_model_data['daily_total'].iloc[split_d:]

daily_pipeline = make_pipeline(n_estimators=200, max_depth=15, min_samples_leaf=3)
daily_pipeline.fit(X_d_train, y_d_train)

y_d_pred = daily_pipeline.predict(X_d_test)
rmse_d = np.sqrt(mean_squared_error(y_d_test, y_d_pred))
mae_d = mean_absolute_error(y_d_test, y_d_pred)
r2_d = r2_score(y_d_test, y_d_pred)
mape_d = mape(y_d_test, y_d_pred)

print(f"\n=== RF Diário (1-step-ahead, teste) ===")
print(f"  Treino: {len(X_d_train)} dias | Teste: {len(X_d_test)} dias "
      f"({X_d_test.index.min().date()} a {X_d_test.index.max().date()})")
print(f"  RMSE: {rmse_d:.2f} | MAE: {mae_d:.2f} | R2: {r2_d:.4f} | MAPE: {mape_d:.2f}%")

feat_imp_d = pd.DataFrame({'feature': daily_features,
                           'importance': daily_pipeline.named_steps['rf'].feature_importances_}) \
               .sort_values('importance', ascending=False)
print(f"\n  Top 10 features:")
for _, r in feat_imp_d.head(10).iterrows():
    print(f"    {r['feature']}: {r['importance']:.4f}")

# Previsão recursiva (mesmas datas do RF horário)
future_days = pd.date_range(start=start_fc, periods=FORECAST_DAYS, freq='D')
monthly_climate = {c: daily.groupby('month')[c].mean().to_dict() for c in climate_cols}
hist_daily = daily['daily_total'].dropna().tail(31).values.tolist()

forecast_daily_list = []
for future_date in future_days:
    row = {
        'dayofweek': future_date.dayofweek, 'month': future_date.month, 'dayofyear': future_date.dayofyear,
        'dow_sin': np.sin(2 * np.pi * future_date.dayofweek / 7),
        'dow_cos': np.cos(2 * np.pi * future_date.dayofweek / 7),
        'month_sin': np.sin(2 * np.pi * future_date.month / 12),
        'month_cos': np.cos(2 * np.pi * future_date.month / 12),
        'dayofyear_sin': np.sin(2 * np.pi * future_date.dayofyear / 365.25),
        'dayofyear_cos': np.cos(2 * np.pi * future_date.dayofyear / 365.25),
    }
    for c in climate_cols:
        row[c] = monthly_climate[c].get(future_date.month, daily[c].mean())
    for lag in LAGS_D:
        row[f'lag_{lag}d'] = hist_daily[-lag] if len(hist_daily) >= lag else hist_daily[0]
    for ma in MAS_D:
        recent = hist_daily[-ma:] if len(hist_daily) >= ma else hist_daily
        row[f'ma_{ma}d'] = np.mean(recent) if recent else 0

    pred_d = daily_pipeline.predict(pd.DataFrame([row], columns=daily_features))[0]
    forecast_daily_list.append(pred_d)
    hist_daily.append(pred_d)
    hist_daily = hist_daily[-60:]

seasonal_forecast = pd.Series(forecast_daily_list, index=future_days)

print(f"\n=== PREVISÃO DIÁRIA COM SAZONALIDADE ({FORECAST_DAYS} dias a partir de {future_days[0].date()}) ===")
for date, val in seasonal_forecast.items():
    print(f"  {date.date()}: {val:.2f}")
print(f"\nTotal: {seasonal_forecast.sum():.2f} | Média diária: {seasonal_forecast.mean():.2f}")
print(f"\nComparação — RF horário recursivo: média={forecast_daily.mean():.2f} | "
      f"RF diário sazonal: média={seasonal_forecast.mean():.2f} "
      f"({(seasonal_forecast.mean() / forecast_daily.mean() - 1) * 100:+.1f}%)")

# Gráficos
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
x = range(len(seasonal_forecast))
axes[0].bar(x, seasonal_forecast.values, color='#4CAF50', edgecolor='white', alpha=0.8, label='RF diário sazonal')
axes[0].bar(x, forecast_daily.reindex(future_days).values, color='#FF9800', edgecolor='white', alpha=0.5,
            width=0.4, label='RF horário recursivo')
axes[0].set_xticks(x)
axes[0].set_xticklabels([d.strftime('%m-%d') for d in future_days], rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diário')
axes[0].set_title(f'Previsão {FORECAST_DAYS} Dias: Diário Sazonal vs Horário Recursivo')
axes[0].legend()

hist_monthly = daily.groupby(daily.index.to_period('M'))['daily_total'].mean()
prev_label = f"{future_days[0]:%Y-%m}"
axes[1].plot(range(len(hist_monthly)), hist_monthly.values, 'o-', color='#2196F3', label='Histórico (média mensal)')
axes[1].plot([len(hist_monthly)], [seasonal_forecast.mean()], 's', color='#4CAF50', markersize=10,
             label=f'Previsto {prev_label}: {seasonal_forecast.mean():.1f}')
axes[1].set_xticks(range(len(hist_monthly) + 1))
axes[1].set_xticklabels([str(p) for p in hist_monthly.index] + [prev_label], rotation=90, fontsize=7)
axes[1].set_ylabel('Consumo médio diário')
axes[1].set_title('Tendência Mensal: Histórico + Previsão')
axes[1].legend()

axes[2].scatter(y_d_test, y_d_pred, alpha=0.6, s=40)
axes[2].plot([y_d_test.min(), y_d_test.max()], [y_d_test.min(), y_d_test.max()], 'r--', linewidth=1)
axes[2].set_xlabel('Real')
axes[2].set_ylabel('Predito')
axes[2].set_title(f'RF Diário: Real vs Predito (R2={r2_d:.3f}, MAPE={mape_d:.1f}%)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Prophet — previsão com sazonalidade explícita
from prophet import Prophet

YEARLY_FOURIER = 10   # com ~3,5 anos de histórico (antes: 3, adequado a < 2 anos)

prophet_df = daily_total.dropna().rename_axis('ds').reset_index(name='y')
for c in climate_cols:
    prophet_df[c] = daily_climate[c].reindex(prophet_df['ds']).values
    # Prophet não aceita NaN em regressores: preencher com a média histórica do mês
    mes_media = prophet_df.groupby(prophet_df['ds'].dt.month)[c].transform('mean')
    prophet_df[c] = prophet_df[c].fillna(mes_media).fillna(prophet_df[c].mean())

print(f"Dados diários: {len(prophet_df)} dias | "
      f"{prophet_df['ds'].min().date()} a {prophet_df['ds'].max().date()}")

# Feriados nacionais + estaduais SP para todo o histórico e o horizonte
anos = range(prophet_df['ds'].dt.year.min(), future_days[-1].year + 1)
feriados_sp = holidays.Brazil(years=anos, subdiv='SP')
br_holidays = pd.DataFrame({
    'holiday': 'feriado_br',
    'ds': pd.to_datetime(list(feriados_sp.keys())),
    'lower_window': 0,
    'upper_window': 1,
})
print(f"Feriados ({min(anos)}–{max(anos)}): {len(br_holidays)} datas")

def novo_prophet():
    m = Prophet(growth='linear', yearly_seasonality=YEARLY_FOURIER, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode='additive', holidays=br_holidays,
                changepoint_prior_scale=0.01, seasonality_prior_scale=5, holidays_prior_scale=5,
                interval_width=0.95)
    for c in climate_cols:
        m.add_regressor(c)
    return m

# Avaliação no teste (80/20)
split_idx_p = int(len(prophet_df) * (1 - TEST_FRACTION))
train_p, test_p = prophet_df.iloc[:split_idx_p], prophet_df.iloc[split_idx_p:]
m = novo_prophet()
m.fit(train_p)
forecast_test = m.predict(test_p[['ds'] + climate_cols])

rmse_p = np.sqrt(mean_squared_error(test_p['y'], forecast_test['yhat']))
mae_p = mean_absolute_error(test_p['y'], forecast_test['yhat'])
r2_p = r2_score(test_p['y'], forecast_test['yhat'])
mape_p = mape(test_p['y'], forecast_test['yhat'])
print(f"\n=== Prophet (teste: {len(test_p)} dias) ===")
print(f"  RMSE: {rmse_p:.2f} | MAE: {mae_p:.2f} | R2: {r2_p:.4f} | MAPE: {mape_p:.2f}%")

# Previsão nas mesmas datas dos outros modelos (retreina no histórico completo)
m_full = novo_prophet()
m_full.fit(prophet_df)
future = pd.DataFrame({'ds': pd.concat([prophet_df['ds'], pd.Series(future_days)]).drop_duplicates()})
for c in climate_cols:
    monthly_mean = prophet_df.groupby(prophet_df['ds'].dt.month)[c].mean()
    future[c] = future['ds'].dt.month.map(monthly_mean).fillna(prophet_df[c].mean())

forecast_full = m_full.predict(future)
forecast_30_future = forecast_full[forecast_full['ds'].isin(future_days)].copy()

print(f"\n=== PREVISÃO PROPHET ({FORECAST_DAYS} dias a partir de {future_days[0].date()}) ===")
for _, row in forecast_30_future.iterrows():
    print(f"  {row['ds'].date()}: {row['yhat']:.2f}  [{row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}]")
print(f"\nTotal: {forecast_30_future['yhat'].sum():.2f} | Média diária: {forecast_30_future['yhat'].mean():.2f}")

print(f"\n=== COMPARAÇÃO DOS 3 MODELOS ({FORECAST_DAYS} dias) ===")
print(f"  RF horário recursivo: média={forecast_daily.mean():.2f}, total={forecast_daily.sum():.2f}")
print(f"  RF diário sazonal:    média={seasonal_forecast.mean():.2f}, total={seasonal_forecast.sum():.2f}")
print(f"  Prophet:              média={forecast_30_future['yhat'].mean():.2f}, total={forecast_30_future['yhat'].sum():.2f}")

# Gráficos
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
axes[0].plot(prophet_df['ds'], prophet_df['y'], 'o-', color='#2196F3', markersize=2, linewidth=0.6, label='Histórico')
axes[0].plot(forecast_30_future['ds'], forecast_30_future['yhat'], 'o-', color='#4CAF50', markersize=3,
             label='Prophet (previsão)')
axes[0].fill_between(forecast_30_future['ds'], forecast_30_future['yhat_lower'], forecast_30_future['yhat_upper'],
                     alpha=0.2, color='#4CAF50', label='IC 95%')
axes[0].set_ylabel('Consumo diário')
axes[0].set_title(f'Prophet: Histórico + Previsão {FORECAST_DAYS} Dias')
axes[0].legend()

axes[1].plot(forecast_full['ds'], forecast_full['yearly'], color='#FF9800', linewidth=1.5)
axes[1].set_ylabel('Componente anual')
axes[1].set_title('Sazonalidade Anual (Fourier) — Prophet')

xc = np.arange(FORECAST_DAYS)
axes[2].bar(xc - 0.25, forecast_daily.reindex(future_days).values, width=0.25, color='#FF9800', alpha=0.6,
            label='RF horário recursivo')
axes[2].bar(xc, seasonal_forecast.values, width=0.25, color='#2196F3', alpha=0.6, label='RF diário sazonal')
axes[2].bar(xc + 0.25, forecast_30_future['yhat'].values, width=0.25, color='#4CAF50', alpha=0.8, label='Prophet')
axes[2].set_xticks(xc)
axes[2].set_xticklabels([d.strftime('%m-%d') for d in future_days], rotation=45, fontsize=7)
axes[2].set_ylabel('Consumo diário')
axes[2].set_title('Comparação: RF Horário vs RF Diário vs Prophet')
axes[2].legend()
plt.tight_layout()
plt.show()

fig2 = m_full.plot_components(forecast_full)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,SARIMA — ARIMA sazonal
from statsmodels.tsa.statespace.sarimax import SARIMAX

ts = daily_total.copy()   # frequência diária; dias inválidos ficam NaN (o filtro de Kalman trata)
ts = ts.loc[ts.first_valid_index():ts.last_valid_index()].asfreq('D')

print(f"=== ARIMA / SARIMA ===")
print(f"Dados: {ts.notna().sum()} dias válidos | {ts.index.min().date()} a {ts.index.max().date()}")

split_idx = int(len(ts) * (1 - TEST_FRACTION))
train_ts, test_ts = ts.iloc[:split_idx], ts.iloc[split_idx:]
print(f"Treino: {len(train_ts)} | Teste: {len(test_ts)}")

configs = [
    {'name': 'ARIMA(1,1,1)', 'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0)},
    {'name': 'SARIMA(1,1,1)(1,0,1,7)', 'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, 7)},
]

best_aic, best_config = np.inf, None
for cfg in configs:
    print(f"\nTreinando {cfg['name']}...")
    try:
        res = SARIMAX(train_ts, order=cfg['order'], seasonal_order=cfg['seasonal_order'],
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc_test = res.forecast(steps=len(test_ts))
        fc_test.index = test_ts.index
        ok = test_ts.notna()
        rmse_cfg = np.sqrt(mean_squared_error(test_ts[ok], fc_test[ok]))
        mae_cfg = mean_absolute_error(test_ts[ok], fc_test[ok])
        r2_cfg = r2_score(test_ts[ok], fc_test[ok])
        print(f"  AIC: {res.aic:.2f} | RMSE: {rmse_cfg:.2f} | MAE: {mae_cfg:.2f} | R2: {r2_cfg:.4f}")
        if res.aic < best_aic:
            best_aic, best_config = res.aic, cfg
            best_rmse, best_mae, best_r2 = rmse_cfg, mae_cfg, r2_cfg
    except Exception as e:
        print(f"  ERRO: {e}")

print(f"\n=== MELHOR MODELO: {best_config['name']} (AIC={best_aic:.2f}) ===")
print(f"  RMSE: {best_rmse:.2f} | MAE: {best_mae:.2f} | R2: {best_r2:.4f}")

results_full = SARIMAX(ts, order=best_config['order'], seasonal_order=best_config['seasonal_order'],
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

# Passos até cobrir as mesmas datas dos outros modelos
steps = (future_days[-1] - ts.index.max()).days
forecast_obj = results_full.get_forecast(steps=steps)
forecast_sarima_mean = forecast_obj.predicted_mean.reindex(future_days)
forecast_sarima_ci = forecast_obj.conf_int(alpha=0.05).reindex(future_days)

print(f"\n=== PREVISÃO {best_config['name']} ({FORECAST_DAYS} dias a partir de {future_days[0].date()}) ===")
for date in future_days:
    print(f"  {date.date()}: {forecast_sarima_mean[date]:.2f}  "
          f"[{forecast_sarima_ci.loc[date].iloc[0]:.2f} - {forecast_sarima_ci.loc[date].iloc[1]:.2f}]")
print(f"\nTotal: {forecast_sarima_mean.sum():.2f} | Média diária: {forecast_sarima_mean.mean():.2f}")

print(f"\n{'='*60}\n=== COMPARAÇÃO DOS 4 MODELOS ({FORECAST_DAYS} dias) ===\n{'='*60}")
for nome, serie in [('RF horário recursivo', forecast_daily.reindex(future_days)),
                    ('RF diário sazonal', seasonal_forecast),
                    ('Prophet', forecast_30_future.set_index('ds')['yhat']),
                    (best_config['name'], forecast_sarima_mean)]:
    print(f"  {nome:24s} média={serie.mean():.2f}, total={serie.sum():.2f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
axes[0].plot(ts.index, ts.values, 'o-', color='#2196F3', markersize=2, linewidth=0.6, label='Histórico')
axes[0].plot(future_days, forecast_sarima_mean.values, 'o-', color='#9C27B0', markersize=3,
             label=f'{best_config["name"]} (previsão)')
axes[0].fill_between(future_days, forecast_sarima_ci.iloc[:, 0], forecast_sarima_ci.iloc[:, 1],
                     alpha=0.2, color='#9C27B0', label='IC 95%')
axes[0].set_ylabel('Consumo diário')
axes[0].set_title(f'{best_config["name"]}: Histórico + Previsão {FORECAST_DAYS} Dias')
axes[0].legend()

xc = range(FORECAST_DAYS)
axes[1].plot(xc, forecast_daily.reindex(future_days).values, 'o-', color='#FF9800', label='RF horário recursivo', markersize=3)
axes[1].plot(xc, seasonal_forecast.values, 's-', color='#2196F3', label='RF diário sazonal', markersize=3)
axes[1].plot(xc, forecast_30_future['yhat'].values, '^-', color='#4CAF50', label='Prophet', markersize=3)
axes[1].plot(xc, forecast_sarima_mean.values, 'D-', color='#9C27B0', label=best_config['name'], markersize=4)
axes[1].set_xticks(xc)
axes[1].set_xticklabels([d.strftime('%m-%d') for d in future_days], rotation=45, fontsize=7)
axes[1].set_ylabel('Consumo diário')
axes[1].set_title('Comparação: RF Horário vs RF Diário vs Prophet vs SARIMA')
axes[1].legend()

residuals = results_full.resid
axes[2].plot(residuals.index, residuals.values, color='#FF5722', linewidth=0.5)
axes[2].axhline(y=0, color='black', linewidth=0.5)
axes[2].set_ylabel('Resíduo')
axes[2].set_title(f'Resíduos do {best_config["name"]} (dataset completo)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Ensemble dos 4 modelos
# Estratégias: média simples, mediana e média ponderada.
# Pesos: ajuste após comparar as métricas de teste desta execução
# (as métricas mudaram em relação ao rev0 com o novo período e a correção de vazamento).
ENSEMBLE_WEIGHTS = {'RF_Horario': 0.50, 'RF_Diario': 0.10, 'Prophet': 0.10, 'SARIMA': 0.30}

ens_df = pd.DataFrame({
    'RF_Horario': forecast_daily.reindex(future_days),
    'RF_Diario': seasonal_forecast.reindex(future_days),
    'Prophet': forecast_30_future.set_index('ds')['yhat'].reindex(future_days),
    'SARIMA': forecast_sarima_mean.reindex(future_days),
}, index=future_days)
modelos_ens = list(ens_df.columns)

ens_df['Ensemble_Media'] = ens_df[modelos_ens].mean(axis=1)
ens_df['Ensemble_Mediana'] = ens_df[modelos_ens].median(axis=1)
ens_df['Ensemble_Ponderado'] = sum(ens_df[c] * w for c, w in ENSEMBLE_WEIGHTS.items()) / sum(ENSEMBLE_WEIGHTS.values())

# IC combinado: média dos ICs do Prophet e do SARIMA
prophet_ci = forecast_30_future.set_index('ds')[['yhat_lower', 'yhat_upper']].reindex(future_days)
ens_df['IC_lower'] = (prophet_ci['yhat_lower'].values + forecast_sarima_ci.iloc[:, 0].values) / 2
ens_df['IC_upper'] = (prophet_ci['yhat_upper'].values + forecast_sarima_ci.iloc[:, 1].values) / 2

periodo_txt = f"{future_days[0]:%d/%m/%Y}–{future_days[-1]:%d/%m/%Y}"
print(f"{'='*70}\n  ENSEMBLE DOS 4 MODELOS — Previsão {FORECAST_DAYS} dias ({periodo_txt})\n{'='*70}")
print(f"  Pesos (ponderado): {ENSEMBLE_WEIGHTS}")
display(ens_df.round(1))

print(f"\n{'='*70}\n  RESUMO ({FORECAST_DAYS} dias) — média histórica diária: {media_geral:.2f}\n{'='*70}")
resumo = pd.DataFrame({
    'media_dia': ens_df.drop(columns=['IC_lower', 'IC_upper']).mean(),
    'total_periodo': ens_df.drop(columns=['IC_lower', 'IC_upper']).sum(),
})
resumo['vs_media_historica_pct'] = (resumo['media_dia'] / media_geral - 1) * 100
display(resumo.round(2))

# Gráficos
x = range(FORECAST_DAYS)
xt = [d.strftime('%d/%m') for d in future_days]
cores = {'RF_Horario': '#FF9800', 'RF_Diario': '#2196F3', 'Prophet': '#4CAF50', 'SARIMA': '#9C27B0'}
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for c in modelos_ens:
    axes[0].plot(x, ens_df[c].values, 'o-', color=cores[c], markersize=3, alpha=0.6, label=c)
axes[0].plot(x, ens_df['Ensemble_Ponderado'].values, 'k-', linewidth=3, label='Ensemble Ponderado', zorder=5)
axes[0].fill_between(x, ens_df['IC_lower'], ens_df['IC_upper'], alpha=0.1, color='gray', label='IC 95% combinado')
axes[0].set_xticks(x)
axes[0].set_xticklabels(xt, rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diário')
axes[0].set_title(f'Ensemble dos 4 Modelos — Previsão {periodo_txt}')
axes[0].legend(fontsize=8, ncol=2)

axes[1].plot(x, ens_df['Ensemble_Media'].values, 'o-', color='#E91E63', markersize=4, label='Ensemble Média')
axes[1].plot(x, ens_df['Ensemble_Mediana'].values, 's-', color='#00BCD4', markersize=4, label='Ensemble Mediana')
axes[1].plot(x, ens_df['Ensemble_Ponderado'].values, 'D-', color='#3F51B5', markersize=4, label='Ensemble Ponderado')
axes[1].axhline(y=media_geral, color='gray', linestyle='--', alpha=0.5, label=f'Média histórica: {media_geral:.1f}')
axes[1].fill_between(x, ens_df['IC_lower'], ens_df['IC_upper'], alpha=0.1, color='gray')
axes[1].set_xticks(x)
axes[1].set_xticklabels(xt, rotation=45, fontsize=7)
axes[1].set_ylabel('Consumo diário')
axes[1].set_title('Estratégias de Ensemble: Média vs Mediana vs Ponderado')
axes[1].legend(fontsize=8)

for c in modelos_ens:
    axes[2].scatter(ens_df['Ensemble_Ponderado'], ens_df[c], alpha=0.6, s=30, color=cores[c], label=c)
lim = [ens_df[modelos_ens + ['Ensemble_Ponderado']].min().min() - 20,
       ens_df[modelos_ens + ['Ensemble_Ponderado']].max().max() + 20]
axes[2].plot(lim, lim, 'k--', linewidth=1, alpha=0.3)
axes[2].set_xlabel('Ensemble Ponderado')
axes[2].set_ylabel('Previsão individual')
axes[2].set_title('Dispersão: Modelos Individuais vs Ensemble Ponderado')
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

# Salvar (Delta)
ens_save = ens_df.copy()
ens_save.index.name = 'data'
ens_save = ens_save.reset_index()
ens_save['dado_ate'] = str(df.index.max())
ens_save['run_ts'] = pd.Timestamp.now().isoformat()
(spark.createDataFrame(ens_save)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(ENSEMBLE_TABLE))
print(f"\nEnsemble salvo em: {ENSEMBLE_TABLE}")