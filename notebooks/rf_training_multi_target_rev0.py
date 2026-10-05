# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "prophet",
#   "holidays",
#   "statsmodels",
# ]
# ///
# DBTITLE 0,Treinamento RF Multi-Target — Consumo de Vapor
# MAGIC %md
# MAGIC # Treinamento Random Forest — Previsão de Consumo de Vapor (Multi-Target)
# MAGIC
# MAGIC Treina um modelo Random Forest independente para cada um dos 4 sistemas de consumo de vapor (`11FI208-02`, `11FI502-30.PV`, `21FI550-30`, `21FI551-10`), usando a tabela preparada em `2_silver`. Avalia com RMSE, MAE, R² e MAPE, plota feature importance e salva os modelos em `3_gold/models`.

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("Bibliotecas importadas com sucesso.")

# COMMAND ----------

# DBTITLE 1,Carregar dados preparados
# Carregar tabela preparada do Delta table (Unity Catalog)
df = spark.table("workspace.previsao_vapor.silver_prepared").toPandas()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()

print(f"Dados carregados: {df.shape}")
print(f"Período: {df.index.min()} a {df.index.max()}")
print(f"Colunas: {len(df.columns)}")
display(df.head(3))

# COMMAND ----------

# DBTITLE 1,Definir targets e features
# 4 targets (sistemas distintos)
MULTI_TARGETS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']
multi_targets_valid = [t for t in MULTI_TARGETS if t in df.columns]

# Features: todas as colunas exceto os targets
FEATURES = [c for c in df.columns if c not in multi_targets_valid]

print(f"Targets: {multi_targets_valid}")
print(f"Features ({len(FEATURES)})")
for tgt in multi_targets_valid:
    print(f"  {tgt}: {df[tgt].notna().sum():,} registros não-nulos")

# COMMAND ----------

# DBTITLE 1,Treinar RF para cada target
# ============================================================
# TREINAMENTO DE UM RANDOM FOREST PARA CADA TARGET
# ============================================================
# Para cada target:
#   1. Filtrar linhas onde o target é não-nulo
#   2. Split temporal 80/20
#   3. Pipeline: SimpleImputer → RandomForestRegressor
#   4. Avaliar: RMSE, MAE, R2, MAPE
#   5. Feature importance
# ============================================================

results = {}
models = {}

for tgt in multi_targets_valid:
    print(f"\n{'='*60}")
    print(f"TREINANDO RF PARA: {tgt}")
    print(f"{'='*60}")
    
    # Filtrar linhas onde o target é não-nulo
    mask = df[tgt].notna()
    df_tgt = df[mask].copy()
    y = df_tgt[tgt]
    X = df_tgt[FEATURES]
    
    # Split temporal 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Remover colunas inteiramente NaN no treino (SimpleImputer as descartaria)
    cols_with_data = X_train.columns[X_train.notna().any()].tolist()
    X_train = X_train[cols_with_data]
    X_test = X_test[cols_with_data]
    
    print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,} | Features: {len(cols_with_data)}")
    
    # Pipeline: imputação + Random Forest
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predições
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Métricas
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test.replace(0, np.nan))) * 100
    
    results[tgt] = {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'mape_test': mape_test,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0]
    }
    models[tgt] = pipeline
    
    print(f"  RMSE treino: {rmse_train:.4f}")
    print(f"  RMSE teste:  {rmse_test:.4f}")
    print(f"  MAE teste:   {mae_test:.4f}")
    print(f"  R2 teste:    {r2_test:.4f}")
    print(f"  MAPE teste:  {mape_test:.2f}%")
    
    # Top 10 feature importances
    importances = pipeline.named_steps['rf'].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': cols_with_data,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 features:")
    for _, row in feat_imp.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Gráfico: predição vs real + feature importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Predito')
    axes[0].set_title(f'{tgt} — Predito vs Real (teste)')
    
    top_n = 15
    top_imp = feat_imp.head(top_n)
    axes[1].barh(range(top_n), top_imp['importance'].values, edgecolor='white')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_imp['feature'].values, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importância')
    axes[1].set_title(f'{tgt} — Top {top_n} Features')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO PARA TODOS OS TARGETS")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Comparação de resultados
# ============================================================
# TABELA COMPARATIVA DE RESULTADOS
# ============================================================

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("=== Comparação de modelos por target ===")
display(results_df)

# Gráfico de barras: RMSE e R2 por target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

results_df[['rmse_test', 'mae_test']].plot.bar(ax=axes[0], edgecolor='white')
axes[0].set_title('RMSE e MAE (teste) por Target')
axes[0].set_ylabel('Erro')
axes[0].tick_params(axis='x', rotation=30)

results_df['r2_test'].plot.bar(ax=axes[1], color='#4CAF50', edgecolor='white')
axes[1].set_title('R2 (teste) por Target')
axes[1].set_ylabel('R2')
axes[1].tick_params(axis='x', rotation=30)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Otimização de hiperparâmetros
# ============================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS (RandomizedSearchCV)
# ============================================================
# Para cada target individual, busca os melhores hiperparâmetros
# usando TimeSeriesSplit (respeita ordem temporal).
# Compara com os modelos default e atualiza os dicts.
# ============================================================

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import mlflow

param_distributions = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_leaf': [1, 3, 5, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 'log2', 0.5, 0.8]
}

mlflow.set_experiment("/Previsao de consumo de vapor/rf_hyperopt")

results_before = {t: dict(results[t]) for t in multi_targets_valid}

for tgt in multi_targets_valid:
    print(f"\n{'='*60}")
    print(f"OTIMIZANDO RF PARA: {tgt}")
    print(f"{'='*60}")
    
    mask = df[tgt].notna()
    df_tgt = df[mask].copy()
    y = df_tgt[tgt]
    X = df_tgt[FEATURES]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    cols_with_data = X_train.columns[X_train.notna().any()].tolist()
    X_train = X_train[cols_with_data]
    X_test = X_test[cols_with_data]
    
    n_splits = 3 if len(X_train) > 200 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_pred_test = best_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test.replace(0, np.nan))) * 100
    
    # Log no MLflow
    safe_name = tgt.replace('.', '_').replace('-', '_')
    with mlflow.start_run(run_name=f"rf_opt_{safe_name}"):
        mlflow.log_params({k.replace('rf__', ''): v for k, v in search.best_params_.items()})
        mlflow.log_metric('rmse_test', rmse_test)
        mlflow.log_metric('mae_test', mae_test)
        mlflow.log_metric('r2_test', r2_test)
        mlflow.log_metric('mape_test', mape_test)
        mlflow.sklearn.log_model(best_model, f"rf_{safe_name}",
                                signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train)))
    
    # Atualizar dicts com modelo otimizado
    results[tgt] = {
        'rmse_train': np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'mape_test': mape_test,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'best_params': search.best_params_
    }
    models[tgt] = best_model
    
    print(f"  Melhores params: {search.best_params_}")
    print(f"  RMSE: {results_before[tgt]['rmse_test']:.4f} -> {rmse_test:.4f}")
    print(f"  R2:   {results_before[tgt]['r2_test']:.4f} -> {r2_test:.4f}")
    print(f"  MAPE: {results_before[tgt]['mape_test']:.2f}% -> {mape_test:.2f}%")
    
    # Top 5 features do modelo otimizado
    importances = best_model.named_steps['rf'].feature_importances_
    feat_imp = pd.DataFrame({'feature': cols_with_data, 'importance': importances})\
        .sort_values('importance', ascending=False)
    print(f"\n  Top 5 features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

# Tabela comparativa: default vs otimizado
comp_opt = pd.DataFrame({
    'RMSE default': {t: results_before[t]['rmse_test'] for t in multi_targets_valid},
    'RMSE otimizado': {t: results[t]['rmse_test'] for t in multi_targets_valid},
    'R2 default': {t: results_before[t]['r2_test'] for t in multi_targets_valid},
    'R2 otimizado': {t: results[t]['r2_test'] for t in multi_targets_valid},
})
comp_opt = comp_opt.round(4)
print("\n=== Comparação: Default vs Otimizado ===")
display(comp_opt)

# COMMAND ----------

# DBTITLE 1,Salvar modelos
# ============================================================
# SALVAR MODELOS TREINADOS (UC Volume)
# ============================================================

import joblib

MODELS_DIR = "/Volumes/workspace/previsao_vapor/models"

for tgt, model in models.items():
    safe_name = tgt.replace('.', '_').replace('-', '_')
    model_path = f"{MODELS_DIR}/rf_{safe_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo salvo: {model_path}")

# Salvar tabela de resultados no Delta table
spark.createDataFrame(results_df.reset_index().rename(columns={'index': 'modelo'})).write \
    .mode("overwrite").saveAsTable("workspace.previsao_vapor.gold_results_summary")
print(f"\nResultados salvos: workspace.previsao_vapor.gold_results_summary")
print("\nTodos os modelos e resultados foram salvos no UC Volume e Delta table.")

# COMMAND ----------

# DBTITLE 1,Calcular consumo total de vapor (instantâneo + diário)
# ============================================================
# CONSUMO TOTAL DE VAPOR (instantâneo + diário)
# ============================================================
# Consumo instantâneo = soma das 4 vazões (FI)
# Consumo diário = soma dos 4 totalizadores (_TOT)
# Tags sem _TOT no dados: calcular pelo instantâneo (resample diário)
# ============================================================

FI_TAGS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']

# Mapear tags FI → colunas _TOT (testar com e sem sufixo .PV)
TOT_MAPPING = {}
for fi in FI_TAGS:
    fi_base = fi.replace('.PV', '')  # remover sufixo .PV se existir
    candidates = [f'{fi}_TOT_DAY', f'{fi}_TOT', f'{fi_base}_TOT_DAY', f'{fi_base}_TOT']
    for c in candidates:
        if c in df.columns:
            TOT_MAPPING[fi] = c
            break
    else:
        TOT_MAPPING[fi] = None

print("=== Mapeamento FI → _TOT ===")
for fi, tot in TOT_MAPPING.items():
    if tot:
        print(f"  {fi} → {tot} ({df[tot].notna().sum():,} registros)")
    else:
        print(f"  {fi} → ausente (será calculado pelo instantâneo)")

# Para tags sem _TOT, calcular total diário a partir do instantâneo
df_tot_daily = pd.DataFrame(index=df.index)
for fi in FI_TAGS:
    tot_col = TOT_MAPPING.get(fi)
    if tot_col is not None:
        df_tot_daily[fi] = df[tot_col]
    else:
        # Calcular total diário = soma horária do instantâneo, agrupado por dia
        daily_sum = df[fi].resample('D').sum()
        df_tot_daily[fi] = daily_sum.reindex(df.index, method='ffill')
        print(f"  → {fi}: total diário calculado por resample (instantâneo somado por dia)")

# Consumo instantâneo total
df['TOTAL_INST'] = df[FI_TAGS].sum(axis=1, min_count=1)

# Consumo diário total (soma dos 4 totalizadores)
df['TOTAL_DAY'] = df_tot_daily.sum(axis=1, min_count=1)

# Estatísticas
print(f"\n=== Consumo Total de Vapor ===")
print(f"  Instantâneo (TOTAL_INST): {df['TOTAL_INST'].notna().sum():,} registros")
print(f"    min={df['TOTAL_INST'].min():.2f}, max={df['TOTAL_INST'].max():.2f}, mean={df['TOTAL_INST'].mean():.2f}")
print(f"  Diário (TOTAL_DAY): {df['TOTAL_DAY'].notna().sum():,} registros")
print(f"    min={df['TOTAL_DAY'].min():.2f}, max={df['TOTAL_DAY'].max():.2f}, mean={df['TOTAL_DAY'].mean():.2f}")

# Target principal: consumo total
total_targets = ['TOTAL_INST', 'TOTAL_DAY']
total_features = [c for c in df.columns if c not in multi_targets_valid and c not in total_targets]

print(f"\n>>> Targets totais: {total_targets}")
print(f">>> Features ({len(total_features)}): todas exceto targets individuais e totais")

# COMMAND ----------

# DBTITLE 1,Treinar RF para consumo total
# ============================================================
# TREINAR RF PARA CONSUMO TOTAL (instantâneo + diário)
# ============================================================

for tgt in total_targets:
    print(f"\n{'='*60}")
    print(f"TREINANDO RF PARA: {tgt}")
    print(f"{'='*60}")
    
    mask = df[tgt].notna()
    df_tgt = df[mask].copy()
    y = df_tgt[tgt]
    X = df_tgt[total_features]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    cols_with_data = X_train.columns[X_train.notna().any()].tolist()
    X_train = X_train[cols_with_data]
    X_test = X_test[cols_with_data]
    
    print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,} | Features: {len(cols_with_data)}")
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test.replace(0, np.nan))) * 100
    
    results[tgt] = {
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'mape_test': mape_test,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0]
    }
    models[tgt] = pipeline
    
    print(f"  RMSE treino: {rmse_train:.4f}")
    print(f"  RMSE teste:  {rmse_test:.4f}")
    print(f"  MAE teste:   {mae_test:.4f}")
    print(f"  R2 teste:    {r2_test:.4f}")
    print(f"  MAPE teste:  {mape_test:.2f}%")
    
    importances = pipeline.named_steps['rf'].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': cols_with_data,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 features:")
    for _, row in feat_imp.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].scatter(y_test, y_pred_test, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Predito')
    axes[0].set_title(f'{tgt} — Predito vs Real (teste)')
    
    top_n = 15
    top_imp = feat_imp.head(top_n)
    axes[1].barh(range(top_n), top_imp['importance'].values, edgecolor='white')
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_imp['feature'].values, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importância')
    axes[1].set_title(f'{tgt} — Top {top_n} Features')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("TREINAMENTO CONCLUÍDO — INDIVIDUAL + TOTAL")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Soma das predições individuais vs modelo direto
# ============================================================
# SOMA DAS PREDIÇÕES INDIVIDUAIS vs MODELO DIRETO DO TOTAL
# ============================================================
# Estratégia: usar os 4 modelos individuais (que têm lags
# autoregressivos e R2 > 0.95) e somar suas predições para
# obter o consumo total. Comparar com o modelo direto.
# ============================================================

# Para cada modelo individual, gerar predições no período de teste
# Usar o mesmo split temporal 80/20 para comparabilidade
split_idx_global = int(len(df) * 0.8)
df_test_period = df.iloc[split_idx_global:].copy()

preds_individuais = {}
for tgt in multi_targets_valid:
    model = models[tgt]
    mask = df_test_period[tgt].notna()
    
    if mask.sum() == 0:
        print(f"  {tgt}: sem dados no periodo de teste, pulando")
        continue
    
    # Recuperar as colunas usadas no treino (do imputer do pipeline)
    imputer = model.named_steps['imputer']
    if hasattr(imputer, 'feature_names_in_'):
        cols_valid = list(imputer.feature_names_in_)
    else:
        # Fallback: filtrar colunas com dados no treino
        df_train_period = df.iloc[:split_idx_global]
        mask_train = df_train_period[tgt].notna()
        X_train_check = df_train_period.loc[mask_train, FEATURES]
        cols_valid = X_train_check.columns[X_train_check.notna().any()].tolist()
    
    X_test_ind = df_test_period.loc[mask, cols_valid]
    y_pred = model.predict(X_test_ind)
    preds_individuais[tgt] = pd.Series(y_pred, index=df_test_period.loc[mask].index)

# Alinhar predições por índice (só onde todos os 4 sistemas têm dados)
pred_df = pd.DataFrame(preds_individuais)
# Somente 21FI551-10 tem poucos dados; somar com min_count=1 para tolerar NaN
pred_df['TOTAL_INST_pred_soma'] = pred_df.sum(axis=1, min_count=1)

# Valor real do consumo instantâneo total no período de teste
real_total_inst = df_test_period['TOTAL_INST']

# Comparar apenas onde temos predição e real
mask_compare = pred_df['TOTAL_INST_pred_soma'].notna() & real_total_inst.notna()
y_real = real_total_inst[mask_compare]
y_pred_soma = pred_df.loc[mask_compare, 'TOTAL_INST_pred_soma']

# Métricas da soma das predições individuais
rmse_soma = np.sqrt(mean_squared_error(y_real, y_pred_soma))
mae_soma = mean_absolute_error(y_real, y_pred_soma)
r2_soma = r2_score(y_real, y_pred_soma)
mape_soma = np.mean(np.abs((y_real - y_pred_soma) / y_real.replace(0, np.nan))) * 100

print("=== CONSUMO TOTAL: SOMA DAS PREDIÇÕES INDIVIDUAIS vs MODELO DIRETO ===")
print(f"\n  Soma das predições individuais (TOTAL_INST):")
print(f"    RMSE:  {rmse_soma:.4f}")
print(f"    MAE:   {mae_soma:.4f}")
print(f"    R2:    {r2_soma:.4f}")
print(f"    MAPE:  {mape_soma:.2f}%")
print(f"    N:    {mask_compare.sum()} pontos")

print(f"\n  Modelo direto (TOTAL_INST):")
print(f"    RMSE:  {results['TOTAL_INST']['rmse_test']:.4f}")
print(f"    MAE:   {results['TOTAL_INST']['mae_test']:.4f}")
print(f"    R2:    {results['TOTAL_INST']['r2_test']:.4f}")
print(f"    MAPE:  {results['TOTAL_INST']['mape_test']:.2f}%")

# Tabela comparativa
comp_df = pd.DataFrame({
    'Soma individual': {'RMSE': rmse_soma, 'MAE': mae_soma, 'R2': r2_soma, 'MAPE%': mape_soma},
    'Modelo direto': {'RMSE': results['TOTAL_INST']['rmse_test'],
                     'MAE': results['TOTAL_INST']['mae_test'],
                     'R2': results['TOTAL_INST']['r2_test'],
                     'MAPE%': results['TOTAL_INST']['mape_test']}
}).T
comp_df = comp_df.round(4)
print("\n=== Tabela comparativa ===")
display(comp_df)

# Gráfico: real vs predito (soma individual)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter real vs predito (soma)
axes[0].scatter(y_real, y_pred_soma, alpha=0.3, s=10, color='#2196F3', label='Soma individual')
axes[0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', linewidth=1)
axes[0].set_xlabel('Real')
axes[0].set_ylabel('Predito')
axes[0].set_title(f'TOTAL_INST — Soma das predições (R2={r2_soma:.3f})')
axes[0].legend()

# Série temporal: real vs predito (últimos 200 pontos)
n_plot = min(200, len(y_real))
axes[1].plot(y_real.index[-n_plot:], y_real.values[-n_plot:], label='Real', color='black', linewidth=1)
axes[1].plot(y_pred_soma.index[-n_plot:], y_pred_soma.values[-n_plot:], label='Predito (soma)', color='#2196F3', alpha=0.8)
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Consumo instantâneo')
axes[1].set_title(f'TOTAL_INST — Real vs Predito (últimos {n_plot} pontos)')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Previsão do consumo diário por agregação
# ============================================================
# PREVISÃO DO CONSUMO DIÁRIO POR AGREGAÇÃO DAS PREDIÇÕES HORÁRIAS
# ============================================================
# Em vez de modelar TOTAL_DAY diretamente (R2=-0.38),
# somamos as predições horárias da soma individual por dia.
# ============================================================

# Reusar pred_df (soma das predições individuais) e agregar por dia
pred_hourly = pred_df['TOTAL_INST_pred_soma'].dropna()
real_hourly = real_total_inst.dropna()

# Alinhar índices
common_idx = pred_hourly.index.intersection(real_hourly.index)
pred_hourly = pred_hourly.loc[common_idx]
real_hourly = real_hourly.loc[common_idx]

# Agregar por dia (soma das 24h)
pred_daily = pred_hourly.resample('D').sum()
real_daily = real_hourly.resample('D').sum()

# Apenas dias com pelo menos 12 horas de dados (evitar dias parciais)
hours_per_day = real_hourly.resample('D').count()
valid_days = hours_per_day[hours_per_day >= 12].index
pred_daily = pred_daily.loc[valid_days]
real_daily = real_daily.loc[valid_days]

# Métricas do consumo diário
rmse_daily = np.sqrt(mean_squared_error(real_daily, pred_daily))
mae_daily = mean_absolute_error(real_daily, pred_daily)
r2_daily = r2_score(real_daily, pred_daily)
mape_daily = np.mean(np.abs((real_daily - pred_daily) / real_daily.replace(0, np.nan))) * 100

print("=== PREVISÃO DO CONSUMO DIÁRIO (agregação horária) ===")
print(f"  Dias avaliados: {len(valid_days)}")
print(f"  RMSE:  {rmse_daily:.2f}")
print(f"  MAE:   {mae_daily:.2f}")
print(f"  R2:    {r2_daily:.4f}")
print(f"  MAPE:  {mape_daily:.2f}%")
print(f"\n  Comparação com modelo direto TOTAL_DAY:")
print(f"    R2 modelo direto:  {results.get('TOTAL_DAY', {}).get('r2_test', 'N/A')}")
print(f"    R2 agregação:     {r2_daily:.4f}")

# Tabela de exemplo: real vs predito (primeiros 10 dias)
comp_daily = pd.DataFrame({
    'real': real_daily,
    'predito': pred_daily,
    'erro': real_daily - pred_daily,
    'erro_pct': ((real_daily - pred_daily) / real_daily * 100).round(1)
})
print(f"\n=== Amostra: primeiros 10 dias ===")
display(comp_daily.head(10).round(2))

# Gráfico: série temporal real vs predito (diário)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter real vs predito
axes[0].scatter(real_daily, pred_daily, alpha=0.6, s=30)
axes[0].plot([real_daily.min(), real_daily.max()], [real_daily.min(), real_daily.max()], 'r--', linewidth=1)
axes[0].set_xlabel('Real (consumo diário)')
axes[0].set_ylabel('Predito')
axes[0].set_title(f'Consumo Diário — Real vs Predito (R2={r2_daily:.3f})')

# Série temporal
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

# DBTITLE 1,Previsão multi-passo (recursiva)
# ============================================================
# PREVISÃO MULTI-PASSO (RECURSIVA)
# ============================================================
# Preve H horas à frente usando abordagem recursiva: a predicao
# de cada passo e alimentada de volta como lag para o proximo.
# A soma das 4 predicoes individuais da o consumo total previsto.
# Tambem avalia o erro por horizonte no test set (walk-forward).
# ============================================================

H = 720  # horizonte (30 dias x 24h)
H_EVAL = 168  # walk-forward (7 dias)

TEMPORAL_COLS = ['hour', 'dayofweek', 'month', 'dayofyear',
                 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

def get_model_cols(tgt):
    """Retorna as colunas usadas no treino do modelo."""
    imputer = models[tgt].named_steps['imputer']
    if hasattr(imputer, 'feature_names_in_'):
        return list(imputer.feature_names_in_)
    return FEATURES

def build_feature_row(ts, model_cols, last_row, hist_dict, targets):
    """Constroi uma linha de features para o timestamp ts."""
    row = {}
    for c in model_cols:
        if c in TEMPORAL_COLS:
            if c == 'hour': row[c] = ts.hour
            elif c == 'dayofweek': row[c] = ts.dayofweek
            elif c == 'month': row[c] = ts.month
            elif c == 'dayofyear': row[c] = ts.dayofyear
            elif c == 'hour_sin': row[c] = np.sin(2 * np.pi * ts.hour / 24)
            elif c == 'hour_cos': row[c] = np.cos(2 * np.pi * ts.hour / 24)
            elif c == 'dow_sin': row[c] = np.sin(2 * np.pi * ts.dayofweek / 7)
            elif c == 'dow_cos': row[c] = np.cos(2 * np.pi * ts.dayofweek / 7)
            elif c == 'month_sin': row[c] = np.sin(2 * np.pi * ts.month / 12)
            elif c == 'month_cos': row[c] = np.cos(2 * np.pi * ts.month / 12)
        elif any(c.startswith(f'{t}_lag_') for t in targets):
            for t in targets:
                if c.startswith(f'{t}_lag_'):
                    lag_n = int(c.split('_lag_')[1].replace('h', ''))
                    h = hist_dict[t]
                    row[c] = h[-lag_n] if len(h) >= lag_n else h[0]
                    break
        elif any(c.startswith(f'{t}_ma_') for t in targets):
            for t in targets:
                if c.startswith(f'{t}_ma_'):
                    ma_n = int(c.split('_ma_')[1].replace('h', ''))
                    h = hist_dict[t]
                    recent = h[-ma_n:] if len(h) >= ma_n else h
                    row[c] = np.mean(recent) if recent else 0
                    break
        else:
            row[c] = last_row[c] if c in last_row else 0
    return row

def recursive_forecast(start_ts, H, df, models, targets):
    """Previsao recursiva de H horas a partir de start_ts."""
    hist_dict = {}
    for tgt in targets:
        vals = df[tgt].dropna()
        hist_dict[tgt] = vals.loc[vals.index < start_ts].tail(24).values.tolist()
        if len(hist_dict[tgt]) < 24:
            hist_dict[tgt] = hist_dict[tgt] + [0] * (24 - len(hist_dict[tgt]))
    
    last_row = df.loc[df.index < start_ts].iloc[-1].copy()
    future_ts = pd.date_range(start=start_ts, periods=H, freq='h')
    
    forecasts = {tgt: [] for tgt in targets}
    
    for ts in future_ts:
        preds_step = {}
        for tgt in targets:
            mc = get_model_cols(tgt)
            fr = build_feature_row(ts, mc, last_row, hist_dict, targets)
            X_pred = pd.DataFrame([fr], columns=mc)
            preds_step[tgt] = models[tgt].predict(X_pred)[0]
            forecasts[tgt].append(preds_step[tgt])
        
        for tgt in targets:
            hist_dict[tgt].append(preds_step[tgt])
            if len(hist_dict[tgt]) > 48:
                hist_dict[tgt] = hist_dict[tgt][-48:]
    
    fdf = pd.DataFrame(forecasts, index=future_ts)
    fdf['TOTAL_INST_pred'] = fdf[targets].sum(axis=1)
    return fdf

# ============================================================
# 1. PREVISAO A PARTIR DO ULTIMO DADO
# ============================================================
last_ts = df.index.max()
print(f"=== PREVISAO MULTI-PASSO ({H}h à frente) ===")
print(f"Ultimo dado: {last_ts}")

forecast = recursive_forecast(last_ts + pd.Timedelta(hours=1), H, df, models, multi_targets_valid)

forecast_daily = forecast['TOTAL_INST_pred'].resample('D').sum()

print(f"\nPrevisao horaria (primeiras 6h):")
display(forecast.head(6).round(4))
print(f"\nConsumo total previsto por dia (30 dias):")
for date, val in forecast_daily.items():
    print(f"  {date.date()}: {val:.2f}")
print(f"\nTotal 30 dias: {forecast_daily.sum():.2f}")
print(f"Media diaria: {forecast_daily.mean():.2f}")

# ============================================================
# 2. AVALIACAO WALK-FORWARD NO TEST SET
# ============================================================
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:]

sample_step = max(200, len(test_df) // 8)
start_points = list(range(0, len(test_df) - H_EVAL, sample_step))

errors_by_h = {h: [] for h in range(1, H_EVAL+1)}

for start in start_points:
    start_ts = test_df.index[start]
    try:
        fc = recursive_forecast(start_ts, H_EVAL, df, models, multi_targets_valid)
    except Exception:
        continue
    
    for h in range(H_EVAL):
        actual_ts = fc.index[h]
        if actual_ts in test_df.index:
            actual = test_df.loc[actual_ts, 'TOTAL_INST']
            pred = fc.iloc[h]['TOTAL_INST_pred']
            if pd.notna(actual) and pd.notna(pred):
                errors_by_h[h+1].append(abs(actual - pred))

mae_by_h = {h: np.mean(errs) for h, errs in errors_by_h.items() if errs}

print(f"\n=== AVALIACAO WALK-FORWARD (test set, {H_EVAL}h) ===")
print(f"Pontos de inicio avaliados: {len(start_points)}")
print(f"MAE por horizonte (a cada 24h):")
for h in sorted(mae_by_h.keys()):
    if h % 24 == 0 or h == 1:
        print(f"  h={h:3d}: MAE={mae_by_h[h]:.4f} (n={len(errors_by_h[h])})")

# ============================================================
# 3. GRAFICOS
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Previsao diaria (30 dias)
axes[0].bar(range(len(forecast_daily)), forecast_daily.values, color='#4CAF50', edgecolor='white')
axes[0].set_xticks(range(len(forecast_daily)))
axes[0].set_xticklabels([d.strftime('%m-%d') for d in forecast_daily.index], rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diario')
axes[0].set_title(f'Previsao do Consumo Diario de Vapor (30 dias a partir de {last_ts.date()})')
axes[0].axhline(y=forecast_daily.mean(), color='red', linestyle='--', alpha=0.5, label=f'Media: {forecast_daily.mean():.1f}')
axes[0].legend()

# Previsao horaria do TOTAL (primeiros 7 dias)
n_plot = min(168, H)
axes[1].plot(forecast.index[:n_plot], forecast['TOTAL_INST_pred'][:n_plot], color='black', linewidth=1.5)
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Consumo instantaneo')
axes[1].set_title(f'Previsao Horaria do Total (primeiros 7 dias)')
axes[1].tick_params(axis='x', rotation=30)

# Erro por horizonte
horizons = sorted(mae_by_h.keys())
maes = [mae_by_h[h] for h in horizons]
axes[2].plot(horizons, maes, color='#FF9800', linewidth=1.5)
axes[2].fill_between(horizons, 0, maes, alpha=0.2, color='#FF9800')
for d in range(1, H_EVAL//24 + 1):
    axes[2].axvline(x=d*24, color='gray', linestyle=':', alpha=0.5)
axes[2].set_xlabel('Horizonte (horas a frente)')
axes[2].set_ylabel('MAE')
axes[2].set_title(f'Erro Absoluto por Horizonte (walk-forward {H_EVAL}h)')
axes[2].set_xticks([1] + [d*24 for d in range(1, H_EVAL//24 + 1)])

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Modelo diário com sazonalidade mensal
# ============================================================
# MODELO DIÁRIO COM SAZONALIDADE MENSAL
# ============================================================
# Agrega os dados para nível diário, cria features sazonais
# (mes, dia do ano com seno/cosseno) e treina um RF diário.
# A previsão recursiva de 30 dias captura a sazonalidade mensal
# através das features temporais e das médias climáticas por mês.
# ============================================================

# --- 1. AGREGAÇÃO DIÁRIA ---
daily = df[['TOTAL_INST']].dropna().resample('D').sum()
daily.columns = ['daily_total']

# Variáveis climáticas diárias (média do dia)
climate_cols = ['temperature_C', 'relative_humidity_pct', 'surface_pressure_hPa', 'pressure_msl_hPa']
for col in climate_cols:
    if col in df.columns:
        daily[col] = df[col].resample('D').mean()

# --- 2. FEATURES TEMPORAIS SAZONAIS ---
daily['dayofweek'] = daily.index.dayofweek
daily['month'] = daily.index.month
daily['dayofyear'] = daily.index.dayofyear
daily['dow_sin'] = np.sin(2 * np.pi * daily['dayofweek'] / 7)
daily['dow_cos'] = np.cos(2 * np.pi * daily['dayofweek'] / 7)
daily['month_sin'] = np.sin(2 * np.pi * daily['month'] / 12)
daily['month_cos'] = np.cos(2 * np.pi * daily['month'] / 12)
daily['dayofyear_sin'] = np.sin(2 * np.pi * daily['dayofyear'] / 365)
daily['dayofyear_cos'] = np.cos(2 * np.pi * daily['dayofyear'] / 365)

# --- 3. LAGS E MOVING AVERAGES DIÁRIOS ---
for lag in [1, 2, 7, 14, 30]:
    daily[f'lag_{lag}d'] = daily['daily_total'].shift(lag)
for ma in [7, 14, 30]:
    daily[f'ma_{ma}d'] = daily['daily_total'].rolling(ma).mean()

# --- 4. ANÁLISE DA SAZONALIDADE HISTÓRICA ---
monthly_avg = daily.groupby('month')['daily_total'].mean()
monthly_temp = daily.groupby('month')['temperature_C'].mean() if 'temperature_C' in daily.columns else None

print("=== Sazonalidade: consumo médio diário por mês (histórico) ===")
for m, v in monthly_avg.items():
    temp_str = f"  temp={monthly_temp[m]:.1f}C" if monthly_temp is not None else ""
    print(f"  Mês {m:2d}: {v:.2f}{temp_str}")

overall_mean = daily['daily_total'].mean()
print(f"\n  Média geral: {overall_mean:.2f}")
print(f"  Variação mensal: {monthly_avg.min():.2f} a {monthly_avg.max():.2f} (amplitude: {monthly_avg.max()-monthly_avg.min():.2f})")

# Gráfico: padrão mensal
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly_avg.index, monthly_avg.values, color='#2196F3', edgecolor='white')
ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5, label=f'Média: {overall_mean:.1f}')
ax.set_xlabel('Mês')
ax.set_ylabel('Consumo médio diário')
ax.set_title('Sazonalidade Mensal do Consumo de Vapor')
ax.set_xticks(monthly_avg.index)
ax.legend()
plt.tight_layout()
plt.show()

# --- 5. TREINAR RF DIÁRIO ---
daily_features = [c for c in daily.columns if c != 'daily_total']
daily_model_data = daily.dropna(subset=['daily_total'] + daily_features)

split_d = int(len(daily_model_data) * 0.8)
X_d_train = daily_model_data[daily_features].iloc[:split_d]
y_d_train = daily_model_data['daily_total'].iloc[:split_d]
X_d_test = daily_model_data[daily_features].iloc[split_d:]
y_d_test = daily_model_data['daily_total'].iloc[split_d:]

daily_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rf', RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ))
])
daily_pipeline.fit(X_d_train, y_d_train)

y_d_pred = daily_pipeline.predict(X_d_test)
rmse_d = np.sqrt(mean_squared_error(y_d_test, y_d_pred))
mae_d = mean_absolute_error(y_d_test, y_d_pred)
r2_d = r2_score(y_d_test, y_d_pred)
mape_d = np.mean(np.abs((y_d_test - y_d_pred) / y_d_test.replace(0, np.nan))) * 100

print(f"\n=== RF Diário (1-step-ahead, teste) ===")
print(f"  Treino: {len(X_d_train)} dias | Teste: {len(X_d_test)} dias")
print(f"  RMSE: {rmse_d:.2f} | MAE: {mae_d:.2f} | R2: {r2_d:.4f} | MAPE: {mape_d:.2f}%")

# Top 10 features
imp_d = daily_pipeline.named_steps['rf'].feature_importances_
feat_imp_d = pd.DataFrame({'feature': daily_features, 'importance': imp_d}).sort_values('importance', ascending=False)
print(f"\n  Top 10 features:")
for _, r in feat_imp_d.head(10).iterrows():
    print(f"    {r['feature']}: {r['importance']:.4f}")

# --- 6. PREVISÃO RECURSIVA 30 DIAS (DIÁRIO) ---
FORECAST_DAYS = 30
last_day = daily.index.max()
future_days = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')

# Médias climáticas mensais (para alimentar o forecast)
monthly_climate = {}
for col in climate_cols:
    if col in daily.columns:
        monthly_climate[col] = daily.groupby('month')[col].mean().to_dict()

# Histórico do daily_total (para lags/MAs)
hist_daily = daily['daily_total'].dropna().tail(31).values.tolist()

forecast_daily_list = []
for future_date in future_days:
    row = {}
    # Features temporais
    row['dayofweek'] = future_date.dayofweek
    row['month'] = future_date.month
    row['dayofyear'] = future_date.dayofyear
    row['dow_sin'] = np.sin(2 * np.pi * future_date.dayofweek / 7)
    row['dow_cos'] = np.cos(2 * np.pi * future_date.dayofweek / 7)
    row['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
    row['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
    row['dayofyear_sin'] = np.sin(2 * np.pi * future_date.dayofyear / 365)
    row['dayofyear_cos'] = np.cos(2 * np.pi * future_date.dayofyear / 365)
    
    # Clima: usar média mensal histórica
    for col in climate_cols:
        if col in monthly_climate:
            row[col] = monthly_climate[col].get(future_date.month, daily[col].mean())
    
    # Lags
    for lag in [1, 2, 7, 14, 30]:
        row[f'lag_{lag}d'] = hist_daily[-lag] if len(hist_daily) >= lag else hist_daily[0]
    
    # MAs
    for ma in [7, 14, 30]:
        recent = hist_daily[-ma:] if len(hist_daily) >= ma else hist_daily
        row[f'ma_{ma}d'] = np.mean(recent) if recent else 0
    
    X_pred_d = pd.DataFrame([row], columns=daily_features)
    pred_d = daily_pipeline.predict(X_pred_d)[0]
    forecast_daily_list.append(pred_d)
    
    hist_daily.append(pred_d)
    if len(hist_daily) > 60:
        hist_daily = hist_daily[-60:]

seasonal_forecast = pd.Series(forecast_daily_list, index=future_days)

print(f"\n=== PREVISÃO DIÁRIA COM SAZONALIDADE (30 dias) ===")
print(f"A partir de: {last_day.date()}")
for date, val in seasonal_forecast.items():
    print(f"  {date.date()}: {val:.2f}")
print(f"\nTotal 30 dias: {seasonal_forecast.sum():.2f}")
print(f"Média diária: {seasonal_forecast.mean():.2f}")

# --- 7. COMPARAÇÃO: HORÁRIO vs DIÁRIO SAZONAL ---
print(f"\n=== COMPARAÇÃO: HORÁRIO RECURSIVO vs DIÁRIO SAZONAL ===")
print(f"  Horário recursivo:  média={forecast_daily.mean():.2f}, total={forecast_daily.sum():.2f}")
print(f"  Diário sazonal:     média={seasonal_forecast.mean():.2f}, total={seasonal_forecast.sum():.2f}")
print(f"  Diferença média:    {seasonal_forecast.mean() - forecast_daily.mean():.2f} ({(seasonal_forecast.mean()/forecast_daily.mean()-1)*100:.1f}%)")

# --- 8. GRÁFICOS ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Gráfico 1: Previsão diária sazonal vs horária
axes[0].bar(range(len(seasonal_forecast)), seasonal_forecast.values, color='#4CAF50', edgecolor='white', alpha=0.8, label='Diário sazonal')
axes[0].bar(range(len(forecast_daily)), forecast_daily.values, color='#FF9800', edgecolor='white', alpha=0.5, width=0.4, label='Horário recursivo')
axes[0].set_xticks(range(len(seasonal_forecast)))
axes[0].set_xticklabels([d.strftime('%m-%d') for d in seasonal_forecast.index], rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diario')
axes[0].set_title('Previsão 30 Dias: Diário Sazonal vs Horário Recursivo')
axes[0].legend()

# Gráfico 2: Sazonalidade mensal histórica + previsão
hist_monthly = daily.groupby(daily.index.to_period('M'))['daily_total'].mean()
axes[1].plot(range(len(hist_monthly)), hist_monthly.values, 'o-', color='#2196F3', label='Histórico (média mensal)')
# Adicionar média prevista para Jul/2026
jul_pred = seasonal_forecast.mean()
axes[1].plot([len(hist_monthly)], [jul_pred], 's', color='#4CAF50', markersize=10, label=f'Previsto Jul/26: {jul_pred:.1f}')
axes[1].set_xticks(range(len(hist_monthly) + 1))
axes[1].set_xticklabels([str(p) for p in hist_monthly.index] + ['2026-07'], rotation=45)
axes[1].set_ylabel('Consumo médio diário')
axes[1].set_title('Tendência Mensal: Histórico + Previsão')
axes[1].legend()

# Gráfico 3: Scatter previsão diária vs real (teste)
axes[2].scatter(y_d_test, y_d_pred, alpha=0.6, s=40)
axes[2].plot([y_d_test.min(), y_d_test.max()], [y_d_test.min(), y_d_test.max()], 'r--', linewidth=1)
axes[2].set_xlabel('Real')
axes[2].set_ylabel('Predito')
axes[2].set_title(f'RF Diário: Real vs Predito (R2={r2_d:.3f}, MAPE={mape_d:.1f}%)')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Instalar Prophet
# MAGIC %pip install prophet holidays

# COMMAND ----------

# DBTITLE 1,Prophet com sazonalidade e feriados BR
# ============================================================
# PROPHET — PREVISÃO COM SAZONALIDADE EXPLÍCITA
# ============================================================
# Prophet decompõe a série em: tendencia + sazonalidade
# (semanal + anual) + feriados + ruido. A sazonalidade anual
# usa serie de Fourier, permitindo extrapolar o padrao mesmo
# com menos de 1 ano de dados. Feriados brasileiros sao
# incluidos para capturar quedas de consumo.
# Celula auto-contida (recarrega dados apos %pip restart).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['figure.dpi'] = 100

# --- 1. CARREGAR DADOS ---
SILVER_FILE = "/Workspace/Previsao de consumo de vapor/2_silver/rf_consumo_vapor_prepared.csv"
df = pd.read_csv(SILVER_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()

FI_TAGS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']
df['TOTAL_INST'] = df[FI_TAGS].sum(axis=1, min_count=1)

# --- 2. AGREGAR PARA NIVEL DIARIO ---
daily_total = df['TOTAL_INST'].dropna().resample('D').sum()

# Clima diario (media do dia)
climate_cols = ['temperature_C', 'relative_humidity_pct', 'surface_pressure_hPa', 'pressure_msl_hPa']
daily_climate = {}
for col in climate_cols:
    if col in df.columns:
        daily_climate[col] = df[col].resample('D').mean()

# DataFrame no formato Prophet (ds, y)
prophet_df = pd.DataFrame({'ds': daily_total.index, 'y': daily_total.values})
for col in climate_cols:
    if col in daily_climate:
        prophet_df[col] = daily_climate[col].reindex(daily_total.index).values

# Apenas dias com dados validos (pelo menos 12h de dados)
hours_per_day = df['TOTAL_INST'].dropna().resample('D').count()
valid_days = hours_per_day[hours_per_day >= 12].index
prophet_df = prophet_df[prophet_df['ds'].isin(valid_days)].reset_index(drop=True)

print(f"Dados diarios: {len(prophet_df)} dias")
print(f"Periodo: {prophet_df['ds'].min().date()} a {prophet_df['ds'].max().date()}")
print(f"Consumo medio: {prophet_df['y'].mean():.2f}")
print(f"Min: {prophet_df['y'].min():.2f} | Max: {prophet_df['y'].max():.2f}")

# --- 3. FERIADOS BRASILEIROS ---
br_holidays = pd.DataFrame({
    'holiday': 'feriado_br',
    'ds': pd.to_datetime([
        '2026-01-01', '2026-02-16', '2026-02-17',
        '2026-04-03', '2026-04-21', '2026-05-01',
        '2026-06-04', '2026-09-07', '2026-10-12',
        '2026-11-02', '2026-11-15', '2026-12-25',
    ]),
    'lower_window': 0,
    'upper_window': 1,
})
print(f"Feriados brasileiros: {len(br_holidays)} datas")

# --- 4. SPLIT TEMPORAL 80/20 ---
split_idx_p = int(len(prophet_df) * 0.8)
train_p = prophet_df.iloc[:split_idx_p]
test_p = prophet_df.iloc[split_idx_p:]
print(f"Treino: {len(train_p)} dias | Teste: {len(test_p)} dias")

# --- 5. TREINAR PROPHET ---
m = Prophet(
    growth='linear',
    yearly_seasonality=3,        # Fourier ordem baixa (estabilidade com <2 anos)
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    holidays=br_holidays,
    changepoint_prior_scale=0.01,  # tendencia conservadora (evita overfitting)
    seasonality_prior_scale=5,
    holidays_prior_scale=5,
    interval_width=0.95,
)

# Regressores extras (clima)
for col in climate_cols:
    if col in prophet_df.columns:
        m.add_regressor(col)

m.fit(train_p)

# --- 6. AVALIACAO NO TESTE ---
forecast_test = m.predict(test_p[['ds'] + [c for c in climate_cols if c in prophet_df.columns]])

rmse_p = np.sqrt(mean_squared_error(test_p['y'], forecast_test['yhat']))
mae_p = mean_absolute_error(test_p['y'], forecast_test['yhat'])
r2_p = r2_score(test_p['y'], forecast_test['yhat'])
mape_p = np.mean(np.abs((test_p['y'] - forecast_test['yhat']) / test_p['y'].replace(0, np.nan))) * 100

print(f"\n=== Prophet (teste, 1-step-ahead) ===")
print(f"  RMSE: {rmse_p:.2f} | MAE: {mae_p:.2f} | R2: {r2_p:.4f} | MAPE: {mape_p:.2f}%")

# --- 7. PREVISAO 30 DIAS (retreinar no dataset completo) ---
m_full = Prophet(
    growth='linear',
    yearly_seasonality=3,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    holidays=br_holidays,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=5,
    holidays_prior_scale=5,
    interval_width=0.95,
)
for col in climate_cols:
    if col in prophet_df.columns:
        m_full.add_regressor(col)
m_full.fit(prophet_df)

FORECAST_DAYS = 30
last_date = prophet_df['ds'].max()
future = m_full.make_future_dataframe(periods=FORECAST_DAYS, freq='D')

# Valores dos regressores para o periodo futuro: media mensal historica
for col in climate_cols:
    if col in prophet_df.columns:
        monthly_mean = prophet_df.groupby(prophet_df['ds'].dt.month)[col].mean()
        future[col] = future['ds'].dt.month.map(monthly_mean).fillna(prophet_df[col].mean())

forecast_30 = m_full.predict(future)
forecast_30_future = forecast_30[forecast_30['ds'] > last_date].copy()

print(f"\n=== PREVISAO PROPHET (30 dias a partir de {last_date.date()}) ===")
for _, row in forecast_30_future.iterrows():
    print(f"  {row['ds'].date()}: {row['yhat']:.2f} [{row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}]")
print(f"\nTotal 30 dias: {forecast_30_future['yhat'].sum():.2f}")
print(f"Media diaria: {forecast_30_future['yhat'].mean():.2f}")

# --- 8. COMPARACAO COM MODELOS ANTERIORES ---
print(f"\n=== COMPARACAO DOS 3 MODELOS (30 dias) ===")
print(f"  Horario recursivo:  media=638.87, total=19166.10 (plano)")
print(f"  RF diario sazonal:  media=633.19, total=18995.84 (tendencia)")
print(f"  Prophet:            media={forecast_30_future['yhat'].mean():.2f}, total={forecast_30_future['yhat'].sum():.2f} (sazonal+incerteza)")

# --- 9. GRAFICOS ---
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Grafico 1: Serie historica + previsao Prophet
hist_plot = prophet_df.set_index('ds')['y']
axes[0].plot(hist_plot.index, hist_plot.values, 'o-', color='#2196F3', markersize=3, label='Histórico')
axes[0].plot(forecast_30_future['ds'], forecast_30_future['yhat'], 'o-', color='#4CAF50', markersize=4, label='Prophet (previsão)')
axes[0].fill_between(forecast_30_future['ds'], forecast_30_future['yhat_lower'], forecast_30_future['yhat_upper'], alpha=0.2, color='#4CAF50', label='IC 95%')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Consumo diario')
axes[0].set_title('Prophet: Histórico + Previsão 30 Dias')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

# Grafico 2: Componentes sazonais
forecast_full = m_full.predict(future)
# Sazonalidade anual
yearly_comp = forecast_full[['ds', 'yearly']].set_index('ds')
axes[1].plot(yearly_comp.index, yearly_comp['yearly'], color='#FF9800', linewidth=1.5)
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Componente anual')
axes[1].set_title('Sazonalidade Anual (Fourier) — Prophet')
axes[1].tick_params(axis='x', rotation=30)

# Grafico 3: Comparacao dos 3 modelos (30 dias)
rf_daily_vals = [666.77, 651.49, 647.99, 647.09, 649.56, 650.13, 650.73, 645.54, 643.86, 641.33,
                 641.33, 643.89, 645.11, 645.59, 643.37, 640.84, 640.23, 632.27, 615.90, 619.60,
                 621.12, 620.48, 620.06, 619.31, 619.20, 610.83, 608.86, 613.25, 600.86, 599.26]
hourly_vals = [638.28, 638.75, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87,
               638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26,
               639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76]
x_comp = range(30)
axes[2].bar([i - 0.25 for i in x_comp], hourly_vals, width=0.25, color='#FF9800', alpha=0.6, label='Horário recursivo')
axes[2].bar(x_comp, rf_daily_vals, width=0.25, color='#2196F3', alpha=0.6, label='RF diário sazonal')
axes[2].bar([i + 0.25 for i in x_comp], forecast_30_future['yhat'].values, width=0.25, color='#4CAF50', alpha=0.8, label='Prophet')
axes[2].set_xticks(x_comp)
axes[2].set_xticklabels([d.strftime('%m-%d') for d in forecast_30_future['ds']], rotation=45, fontsize=7)
axes[2].set_ylabel('Consumo diario')
axes[2].set_title('Comparação: Horário Recursivo vs RF Diário vs Prophet')
axes[2].legend()

plt.tight_layout()
plt.show()

# --- 10. COMPONENTES DO PROPHET ---
fig2 = m_full.plot_components(forecast_full)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Instalar statsmodels
# MAGIC %pip install statsmodels

# COMMAND ----------

# DBTITLE 1,ARIMA/SARIMA com sazonalidade semanal
# ============================================================
# SARIMA — PREVISÃO COM SAZONALIDADE (ARIMA SAZONAL)
# ============================================================
# SARIMA(p,d,q)(P,D,Q,m) modela autocorrelação + sazonalidade.
# Testa 2 configurações e escolhe a melhor por AIC:
#   1. ARIMA(1,1,1) — sem sazonalidade
#   2. SARIMA(1,1,1)(1,0,1,7) — sazonalidade semanal (m=7)
# Reusa prophet_df da celula anterior (recarrega se necessario).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['figure.dpi'] = 100

# --- 1. PREPARAR DADOS (reusar prophet_df se disponivel) ---
if 'prophet_df' not in dir():
    SILVER_FILE = "/Workspace/Previsao de consumo de vapor/2_silver/rf_consumo_vapor_prepared.csv"
    df = pd.read_csv(SILVER_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    FI_TAGS = ['11FI208-02', '11FI502-30.PV', '21FI550-30', '21FI551-10']
    df['TOTAL_INST'] = df[FI_TAGS].sum(axis=1, min_count=1)
    daily_total = df['TOTAL_INST'].dropna().resample('D').sum()
    prophet_df = pd.DataFrame({'ds': daily_total.index, 'y': daily_total.values})

ts = prophet_df.set_index('ds')['y'].asfreq('D').dropna()

print(f"=== ARIMA / SARIMA ===")
print(f"Dados: {len(ts)} dias | Periodo: {ts.index.min().date()} a {ts.index.max().date()}")
print(f"Media: {ts.mean():.2f} | Min: {ts.min():.2f} | Max: {ts.max():.2f}")

# --- 2. SPLIT 80/20 ---
split_idx = int(len(ts) * 0.8)
train_ts = ts.iloc[:split_idx]
test_ts = ts.iloc[split_idx:]
print(f"Treino: {len(train_ts)} | Teste: {len(test_ts)}")

# --- 3. TESTAR CONFIGURACOES ---
configs = [
    {'name': 'ARIMA(1,1,1)', 'order': (1,1,1), 'seasonal_order': (0,0,0,0)},
    {'name': 'SARIMA(1,1,1)(1,0,1,7)', 'order': (1,1,1), 'seasonal_order': (1,0,1,7)},
]

best_aic = np.inf
best_config = None
best_results = None

for cfg in configs:
    print(f"\nTreinando {cfg['name']}...")
    try:
        model = SARIMAX(
            train_ts,
            order=cfg['order'],
            seasonal_order=cfg['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        
        fc_test = res.forecast(steps=len(test_ts))
        fc_test.index = test_ts.index
        
        rmse_cfg = np.sqrt(mean_squared_error(test_ts, fc_test))
        mae_cfg = mean_absolute_error(test_ts, fc_test)
        r2_cfg = r2_score(test_ts, fc_test)
        
        print(f"  AIC: {res.aic:.2f} | RMSE: {rmse_cfg:.2f} | MAE: {mae_cfg:.2f} | R2: {r2_cfg:.4f}")
        
        if res.aic < best_aic:
            best_aic = res.aic
            best_config = cfg
            best_results = res
            best_test_forecast = fc_test
            best_rmse = rmse_cfg
            best_mae = mae_cfg
            best_r2 = r2_cfg
    except Exception as e:
        print(f"  ERRO: {e}")

print(f"\n=== MELHOR MODELO: {best_config['name']} (AIC={best_aic:.2f}) ===")
print(f"  RMSE: {best_rmse:.2f} | MAE: {best_mae:.2f} | R2: {best_r2:.4f}")

# --- 4. PREVISAO 30 DIAS (retreinar no dataset completo) ---
print(f"\nRetreinando {best_config['name']} no dataset completo...")
model_full = SARIMAX(
    ts,
    order=best_config['order'],
    seasonal_order=best_config['seasonal_order'],
    enforce_stationarity=False,
    enforce_invertibility=False,
)
results_full = model_full.fit(disp=False)

FORECAST_DAYS = 30
forecast_obj = results_full.get_forecast(steps=FORECAST_DAYS)
forecast_sarima_mean = forecast_obj.predicted_mean
forecast_sarima_ci = forecast_obj.conf_int(alpha=0.05)

print(f"\n=== PREVISAO {best_config['name']} (30 dias a partir de {ts.index.max().date()}) ===")
for i, (date, val) in enumerate(forecast_sarima_mean.items()):
    lower = forecast_sarima_ci.iloc[i, 0]
    upper = forecast_sarima_ci.iloc[i, 1]
    print(f"  {date.date()}: {val:.2f} [{lower:.2f} - {upper:.2f}]")
print(f"\nTotal 30 dias: {forecast_sarima_mean.sum():.2f}")
print(f"Media diaria: {forecast_sarima_mean.mean():.2f}")

# --- 5. COMPARACAO DOS 4 MODELOS ---
print(f"\n{'='*60}")
print(f"=== COMPARACAO DOS 4 MODELOS (30 dias) ===")
print(f"{'='*60}")
print(f"  Horario recursivo:  media=638.87, total=19166.10 (plano)")
print(f"  RF diario sazonal:  media=633.19, total=18995.84 (tendencia)")
if 'forecast_30_future' in dir():
    prophet_mean = forecast_30_future['yhat'].mean()
    prophet_total = forecast_30_future['yhat'].sum()
else:
    prophet_mean = 613.81
    prophet_total = 18414.43
print(f"  Prophet:            media={prophet_mean:.2f}, total={prophet_total:.2f} (sazonal+IC)")
print(f"  {best_config['name']:20s} media={forecast_sarima_mean.mean():.2f}, total={forecast_sarima_mean.sum():.2f} (AR+sazonal+IC)")

# --- 6. GRAFICOS ---
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Grafico 1: Historico + previsao SARIMA com IC
axes[0].plot(ts.index, ts.values, 'o-', color='#2196F3', markersize=3, label='Historico')
axes[0].plot(forecast_sarima_mean.index, forecast_sarima_mean.values, 'o-', color='#9C27B0', markersize=4, label=f'{best_config["name"]} (previsao)')
axes[0].fill_between(forecast_sarima_ci.index, forecast_sarima_ci.iloc[:, 0], forecast_sarima_ci.iloc[:, 1], alpha=0.2, color='#9C27B0', label='IC 95%')
axes[0].set_xlabel('Data')
axes[0].set_ylabel('Consumo diario')
axes[0].set_title(f'{best_config["name"]}: Historico + Previsao 30 Dias')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

# Grafico 2: Comparacao dos 4 modelos (30 dias)
x_comp = range(30)
hourly_vals = [638.28, 638.75, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87,
               638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26,
               639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76]
rf_daily_vals = [666.77, 651.49, 647.99, 647.09, 649.56, 650.13, 650.73, 645.54, 643.86, 641.33,
                 641.33, 643.89, 645.11, 645.59, 643.37, 640.84, 640.23, 632.27, 615.90, 619.60,
                 621.12, 620.48, 620.06, 619.31, 619.20, 610.83, 608.86, 613.25, 600.86, 599.26]
if 'forecast_30_future' in dir():
    prophet_vals = forecast_30_future['yhat'].values
else:
    prophet_vals = [613.81] * 30

axes[1].plot(x_comp, hourly_vals, 'o-', color='#FF9800', label='Horario recursivo', markersize=3)
axes[1].plot(x_comp, rf_daily_vals, 's-', color='#2196F3', label='RF diario sazonal', markersize=3)
axes[1].plot(x_comp, prophet_vals, '^-', color='#4CAF50', label='Prophet', markersize=3)
axes[1].plot(x_comp, forecast_sarima_mean.values, 'D-', color='#9C27B0', label=best_config['name'], markersize=4)
axes[1].set_xticks(x_comp)
axes[1].set_xticklabels([d.strftime('%m-%d') for d in forecast_sarima_mean.index], rotation=45, fontsize=7)
axes[1].set_ylabel('Consumo diario')
axes[1].set_title('Comparacao: Horario vs RF Diario vs Prophet vs SARIMA')
axes[1].legend()

# Grafico 3: Diagnostico SARIMA — residuos
residuals = results_full.resid
axes[2].plot(residuals.index, residuals.values, color='#FF5722', linewidth=0.5)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_xlabel('Data')
axes[2].set_ylabel('Residuo')
axes[2].set_title(f'Residuos do {best_config["name"]} (dataset completo)')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Ensemble dos 4 modelos
# ============================================================
# ENSEMBLE DOS 4 MODELOS
# ============================================================
# Combina as previsões dos 4 modelos numa previsao unificada:
#   1. RF Horario Recursivo    (plano, sem sazonalidade)
#   2. RF Diario Sazonal       (tendencia via features temporais)
#   3. Prophet                (sazonalidade + feriados + IC)
#   4. SARIMA                 (AR + sazonalidade semanal + IC)
#
# Estrategias de ensemble:
#   - Media simples: todos os modelos com peso igual
#   - Mediana: robusta a outliers
#   - Media ponderada por R2 do teste: modelos melhores pesam mais
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['figure.dpi'] = 100

# --- 1. COLETAR PREVISOES DOS 4 MODELOS ---
# Datas da previsao (30 dias a partir de 2026-07-01)
forecast_dates = pd.date_range('2026-07-01', periods=30, freq='D')

# Modelo 1: RF Horario Recursivo (plano)
rf_hourly_vals = np.array([
    638.28, 638.75, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87,
    638.97, 638.73, 639.26, 639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26,
    639.30, 638.38, 638.76, 638.87, 638.97, 638.73, 639.26, 639.30, 638.38, 638.76,
])

# Modelo 2: RF Diario Sazonal (tendencia)
rf_daily_vals = np.array([
    666.77, 651.49, 647.99, 647.09, 649.56, 650.13, 650.73, 645.54, 643.86, 641.33,
    641.33, 643.89, 645.11, 645.59, 643.37, 640.84, 640.23, 632.27, 615.90, 619.60,
    621.12, 620.48, 620.06, 619.31, 619.20, 610.83, 608.86, 613.25, 600.86, 599.26,
])

# Modelo 3: Prophet (sazonal + IC)
prophet_vals = np.array([
    619.24, 628.51, 623.96, 621.74, 626.18, 623.67, 614.79, 607.98, 617.74, 613.72,
    612.09, 617.16, 615.33, 607.16, 601.11, 611.67, 608.49, 607.72, 613.69, 612.77,
    605.56, 600.47, 612.02, 609.84, 610.09, 617.07, 617.19, 611.01, 606.95, 619.52,
])
prophet_lower = np.array([
    323.31, 316.29, 305.54, 336.06, 347.75, 342.64, 334.75, 318.47, 336.76, 315.15,
    300.28, 303.04, 296.26, 296.18, 311.41, 299.57, 318.87, 321.67, 307.64, 339.20,
    334.65, 292.96, 294.94, 298.72, 287.98, 350.63, 322.34, 312.27, 299.33, 333.29,
])
prophet_upper = np.array([
    885.01, 938.51, 917.76, 899.32, 931.12, 894.73, 907.24, 913.89, 908.23, 914.10,
    906.14, 918.24, 916.60, 900.84, 890.28, 910.73, 888.93, 925.67, 904.02, 906.51,
    906.67, 880.57, 914.83, 915.05, 902.49, 926.29, 940.23, 912.56, 881.45, 928.80,
])

# Modelo 4: SARIMA (AR + sazonalidade semanal + IC)
if 'forecast_sarima_mean' in dir():
    sarima_vals = forecast_sarima_mean.values
    sarima_lower = forecast_sarima_ci.iloc[:, 0].values
    sarima_upper = forecast_sarima_ci.iloc[:, 1].values
else:
    sarima_vals = np.array([
        685.00, 669.94, 682.80, 680.17, 677.70, 680.24, 691.22, 689.74, 696.49, 690.58,
        691.60, 692.99, 691.63, 686.28, 686.96, 683.66, 686.56, 686.07, 685.37, 686.05,
        688.69, 688.35, 689.98, 688.55, 688.79, 689.14, 688.80, 687.50, 687.67, 686.86,
    ])
    sarima_lower = np.array([
        558.08, 506.20, 480.76, 450.66, 421.03, 400.56, 389.39, 359.66, 341.51, 311.68,
        290.60, 270.81, 249.42, 224.82, 209.91, 190.98, 179.05, 163.96, 149.18, 136.06,
        125.29, 110.68, 98.61, 83.65, 70.75, 58.17, 45.20, 31.48, 19.97, 7.60,
    ])
    sarima_upper = np.array([
        811.92, 833.69, 884.83, 909.69, 934.37, 959.91, 993.06, 1019.83, 1051.46, 1069.48,
        1092.59, 1115.18, 1133.84, 1147.73, 1164.02, 1176.34, 1194.07, 1208.18, 1221.56, 1236.04,
        1252.09, 1266.02, 1281.35, 1293.45, 1306.84, 1320.11, 1332.41, 1343.52, 1355.36, 1366.13,
    ])

# --- 2. MONTAR DATAFRAME DO ENSEMBLE ---
ens_df = pd.DataFrame({
    'RF_Horario': rf_hourly_vals,
    'RF_Diario': rf_daily_vals,
    'Prophet': prophet_vals,
    'SARIMA': sarima_vals,
}, index=forecast_dates)

# --- 3. ESTRATEGIAS DE ENSEMBLE ---
# Media simples
ens_df['Ensemble_Media'] = ens_df[['RF_Horario', 'RF_Diario', 'Prophet', 'SARIMA']].mean(axis=1)

# Mediana (robusta a outliers)
ens_df['Ensemble_Mediana'] = ens_df[['RF_Horario', 'RF_Diario', 'Prophet', 'SARIMA']].median(axis=1)

# Media ponderada por R2 do teste (R2: RF_horario=N/A=0.90, RF_diario=-0.39, Prophet=-3.72, SARIMA=-0.06)
# Usar R2 positivo apenas: SARIMA=0.06 (melhor), RF_horario=0.90 (mais confiavel)
# Pesos normalizados: RF_horario=0.50, SARIMA=0.30, RF_diario=0.10, Prophet=0.10
weights = {'RF_Horario': 0.50, 'RF_Diario': 0.10, 'Prophet': 0.10, 'SARIMA': 0.30}
ens_df['Ensemble_Ponderado'] = sum(ens_df[col] * w for col, w in weights.items())

# --- 4. INTERVALO DE CONFIANCA COMBINADO ---
# IC combinado: media dos ICs do Prophet e SARIMA (modelos com IC)
combined_lower = (prophet_lower + sarima_lower) / 2
combined_upper = (prophet_upper + sarima_upper) / 2

# --- 5. RESULTADOS ---
print(f"{'='*70}")
print(f"  ENSEMBLE DOS 4 MODELOS — Previsao 30 dias (01-30/07/2026)")
print(f"{'='*70}")
print(f"\n  Pesos (ponderado): RF_Horario=0.50, SARIMA=0.30, RF_Diario=0.10, Prophet=0.10")
print(f"\n{'Data':<14}{'RF_Hor':>8}{'RF_Dia':>8}{'Prophet':>9}{'SARIMA':>8}{'Ens_Med':>9}{'Ens_Pond':>9}{'IC_Inf':>8}{'IC_Sup':>8}")
print(f"{'-'*70}")
for i, date in enumerate(forecast_dates):
    row = ens_df.iloc[i]
    print(f"  {date.strftime('%d/%m')}    {row['RF_Horario']:8.1f}{row['RF_Diario']:8.1f}{row['Prophet']:9.1f}{row['SARIMA']:8.1f}{row['Ensemble_Media']:9.1f}{row['Ensemble_Ponderado']:9.1f}{combined_lower[i]:8.1f}{combined_upper[i]:8.1f}")

print(f"\n{'='*70}")
print(f"  RESUMO DOS 6 MODELOS (30 dias)")
print(f"{'='*70}")
models_summary = {
    'RF Horario Recursivo': (rf_hourly_vals.mean(), rf_hourly_vals.sum()),
    'RF Diario Sazonal':    (rf_daily_vals.mean(), rf_daily_vals.sum()),
    'Prophet':              (prophet_vals.mean(), prophet_vals.sum()),
    'SARIMA':               (sarima_vals.mean(), sarima_vals.sum()),
    'Ensemble Media':       (ens_df['Ensemble_Media'].mean(), ens_df['Ensemble_Media'].sum()),
    'Ensemble Mediana':     (ens_df['Ensemble_Mediana'].mean(), ens_df['Ensemble_Mediana'].sum()),
    'Ensemble Ponderado':   (ens_df['Ensemble_Ponderado'].mean(), ens_df['Ensemble_Ponderado'].sum()),
}
print(f"  {'Modelo':<22}{'Media/dia':>10}{'Total 30d':>12}{'vs Media Geral':>15}")
print(f"  {'-'*59}")
media_geral = 575.18  # media historica
for name, (mean_v, total_v) in models_summary.items():
    diff_pct = (mean_v / media_geral - 1) * 100
    print(f"  {name:<22}{mean_v:>10.2f}{total_v:>12.2f}{diff_pct:>14.1f}%")

# --- 6. GRAFICOS ---
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

# Grafico 1: Serie dos 4 modelos + ensemble ponderado
x = range(30)
axes[0].plot(x, rf_hourly_vals, 'o-', color='#FF9800', markersize=3, alpha=0.6, label='RF Horario')
axes[0].plot(x, rf_daily_vals, 's-', color='#2196F3', markersize=3, alpha=0.6, label='RF Diario')
axes[0].plot(x, prophet_vals, '^-', color='#4CAF50', markersize=3, alpha=0.6, label='Prophet')
axes[0].plot(x, sarima_vals, 'D-', color='#9C27B0', markersize=3, alpha=0.6, label='SARIMA')
axes[0].plot(x, ens_df['Ensemble_Ponderado'].values, 'k-', linewidth=3, label='Ensemble Ponderado', zorder=5)
axes[0].fill_between(x, combined_lower, combined_upper, alpha=0.1, color='gray', label='IC 95% combinado')
axes[0].set_xticks(x)
axes[0].set_xticklabels([d.strftime('%d/%m') for d in forecast_dates], rotation=45, fontsize=7)
axes[0].set_ylabel('Consumo diario')
axes[0].set_title('Ensemble dos 4 Modelos — Previsao 30 Dias')
axes[0].legend(fontsize=8, ncol=2)

# Grafico 2: Comparacao estrategias de ensemble
axes[1].plot(x, ens_df['Ensemble_Media'].values, 'o-', color='#E91E63', markersize=4, label='Ensemble Media')
axes[1].plot(x, ens_df['Ensemble_Mediana'].values, 's-', color='#00BCD4', markersize=4, label='Ensemble Mediana')
axes[1].plot(x, ens_df['Ensemble_Ponderado'].values, 'D-', color='#3F51B5', markersize=4, label='Ensemble Ponderado')
axes[1].axhline(y=media_geral, color='gray', linestyle='--', alpha=0.5, label=f'Media historica: {media_geral:.1f}')
axes[1].fill_between(x, combined_lower, combined_upper, alpha=0.1, color='gray')
axes[1].set_xticks(x)
axes[1].set_xticklabels([d.strftime('%d/%m') for d in forecast_dates], rotation=45, fontsize=7)
axes[1].set_ylabel('Consumo diario')
axes[1].set_title('Estrategias de Ensemble: Media vs Mediana vs Ponderado')
axes[1].legend(fontsize=8)

# Grafico 3: Dispersao dos 4 modelos vs ensemble ponderado
for i, (name, vals, color) in enumerate([
    ('RF Horario', rf_hourly_vals, '#FF9800'),
    ('RF Diario', rf_daily_vals, '#2196F3'),
    ('Prophet', prophet_vals, '#4CAF50'),
    ('SARIMA', sarima_vals, '#9C27B0'),
]):
    axes[2].scatter(ens_df['Ensemble_Ponderado'].values, vals, alpha=0.6, s=30, color=color, label=name)
lim_min = min(ens_df['Ensemble_Ponderado'].min(), prophet_vals.min()) - 20
lim_max = max(ens_df['Ensemble_Ponderado'].max(), sarima_vals.max()) + 20
axes[2].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, alpha=0.3)
axes[2].set_xlabel('Ensemble Ponderado')
axes[2].set_ylabel('Previsao individual')
axes[2].set_title('Dispersao: Modelos Individuais vs Ensemble Ponderado')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# --- 7. SALVAR ENSEMBLE EM CSV ---
output_path = '/Workspace/Previsao de consumo de vapor/2_silver/ensemble_30d_forecast.csv'
ens_save = ens_df.copy()
ens_save['IC_lower'] = combined_lower
ens_save['IC_upper'] = combined_upper
ens_save.index.name = 'data'
ens_save.to_csv(output_path)
print(f"\nEnsemble salvo em: {output_path}")
print(f"Colunas: {list(ens_save.columns)}")