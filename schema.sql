-- ============================================================
-- Schema: workspace.previsao_vapor
-- Unity Catalog - DDLs das tabelas Delta
-- Gerado automaticamente via SHOW CREATE TABLE
-- Atualizado em: 2026-10-06
-- ============================================================

CREATE CATALOG IF NOT EXISTS workspace;
CREATE SCHEMA IF NOT EXISTS workspace.previsao_vapor;

-- ============================================================
-- BRONZE
-- ============================================================

-- Tabela: workspace.previsao_vapor.bronze_clima
-- Linhas: 48.168 | Dados meteorologicos (2021-Jun 2026)
CREATE TABLE workspace.previsao_vapor.bronze_clima (
  datetime_local TIMESTAMP,
  temperature_C DOUBLE,
  relative_humidity_pct INT,
  surface_pressure_hPa DOUBLE,
  pressure_msl_hPa DOUBLE)
USING delta
COMMENT 'The table contains climate data recorded over time, focusing on various atmospheric conditions. Use cases include analyzing temperature trends, humidity levels, and pressure variations. This data can help in weather forecasting and understanding local climate behavior.'
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.bronze_processo
-- Linhas: 308.896.910 | Dados de processo em formato long (tag, ts, value)
CREATE TABLE workspace.previsao_vapor.bronze_processo (
  tag STRING COLLATE UTF8_BINARY,
  ts TIMESTAMP,
  value DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- ============================================================
-- SILVER
-- ============================================================

-- Tabela: workspace.previsao_vapor.silver_prepared
-- Linhas: 30.421 | Dataset preparado para Random Forest (90 colunas)
CREATE TABLE workspace.previsao_vapor.silver_prepared (
  datetime TIMESTAMP,
  `01FC466-02.PV` DOUBLE,
  `01FC466-02_TOT_DAY` DOUBLE,
  `01FI980-01.PV` DOUBLE,
  `01FI980-01_TOT_DAY` DOUBLE,
  `01SF380-90.PV` DOUBLE,
  `11AC208-66` DOUBLE,
  `11AC502-01` DOUBLE,
  `11FC208-106.PV` DOUBLE,
  `11FC208-138.PV` DOUBLE,
  `11FC502-01.PV` DOUBLE,
  `11FC502-04.PV` DOUBLE,
  `11FI208-02` DOUBLE,
  `11FI208-02_TOT_DAY` DOUBLE,
  `11FI502-30.PV` DOUBLE,
  `11FI502-30_TOT_DAY` DOUBLE,
  `11LC202-02` DOUBLE,
  `11LC502-31.PV` DOUBLE,
  `11LI208-38` DOUBLE,
  `11PC208-01` DOUBLE,
  `11PC502-35` DOUBLE,
  `11TI202-14` DOUBLE,
  `21AC550-17` DOUBLE,
  `21FC550-69.PV` DOUBLE,
  `21FC550-72.PV` DOUBLE,
  `21FI533-05_TOT_DAY` DOUBLE,
  `21FI533-08_TOT_DAY` DOUBLE,
  `21FI550-30` DOUBLE,
  `21FI551-10` DOUBLE,
  `21FI551-10_TOT` DOUBLE,
  `21LC552-03` DOUBLE,
  `21MASTER_PRESS` DOUBLE,
  `21PC550-35` DOUBLE,
  `21PI550-15` DOUBLE,
  `21PI550-84` DOUBLE,
  `21PI550-95` DOUBLE,
  `21TI550-19` DOUBLE,
  `21TI550-91B` DOUBLE,
  `31LC800-21_PV` DOUBLE,
  `31PIC800-40_PV` DOUBLE,
  temperature_C DOUBLE,
  relative_humidity_pct DOUBLE,
  surface_pressure_hPa DOUBLE,
  pressure_msl_hPa DOUBLE,
  hour INT,
  dayofweek INT,
  month INT,
  dayofyear INT,
  hour_sin DOUBLE,
  hour_cos DOUBLE,
  dow_sin DOUBLE,
  dow_cos DOUBLE,
  month_sin DOUBLE,
  month_cos DOUBLE,
  `11FI208-02_lag_1h` DOUBLE,
  `11FI208-02_lag_3h` DOUBLE,
  `11FI208-02_lag_6h` DOUBLE,
  `11FI208-02_lag_12h` DOUBLE,
  `11FI208-02_lag_24h` DOUBLE,
  `11FI208-02_ma_3h` DOUBLE,
  `11FI208-02_ma_6h` DOUBLE,
  `11FI208-02_ma_12h` DOUBLE,
  `11FI208-02_ma_24h` DOUBLE,
  `11FI502-30.PV_lag_1h` DOUBLE,
  `11FI502-30.PV_lag_3h` DOUBLE,
  `11FI502-30.PV_lag_6h` DOUBLE,
  `11FI502-30.PV_lag_12h` DOUBLE,
  `11FI502-30.PV_lag_24h` DOUBLE,
  `11FI502-30.PV_ma_3h` DOUBLE,
  `11FI502-30.PV_ma_6h` DOUBLE,
  `11FI502-30.PV_ma_12h` DOUBLE,
  `11FI502-30.PV_ma_24h` DOUBLE,
  `21FI550-30_lag_1h` DOUBLE,
  `21FI550-30_lag_3h` DOUBLE,
  `21FI550-30_lag_6h` DOUBLE,
  `21FI550-30_lag_12h` DOUBLE,
  `21FI550-30_lag_24h` DOUBLE,
  `21FI550-30_ma_3h` DOUBLE,
  `21FI550-30_ma_6h` DOUBLE,
  `21FI550-30_ma_12h` DOUBLE,
  `21FI550-30_ma_24h` DOUBLE,
  `21FI551-10_lag_1h` DOUBLE,
  `21FI551-10_lag_3h` DOUBLE,
  `21FI551-10_lag_6h` DOUBLE,
  `21FI551-10_lag_12h` DOUBLE,
  `21FI551-10_lag_24h` DOUBLE,
  `21FI551-10_ma_3h` DOUBLE,
  `21FI551-10_ma_6h` DOUBLE,
  `21FI551-10_ma_12h` DOUBLE,
  `21FI551-10_ma_24h` DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.silver_diario
-- Linhas: 1.361 | Consumo diario de gas e biomassa (17 colunas)
CREATE TABLE workspace.previsao_vapor.silver_diario (
  datetime TIMESTAMP,
  C208_gas DOUBLE,
  C208_ar DOUBLE,
  C502_gas DOUBLE,
  C502_ar DOUBLE,
  C208_vapor DOUBLE,
  C502_vapor DOUBLE,
  C550_ar DOUBLE,
  C550_gas DOUBLE,
  C550_vapor DOUBLE,
  BIO_vapor DOUBLE,
  vapor_gas_total DOUBLE,
  vapor_biomassa DOUBLE,
  demanda_vapor_total DOUBLE,
  gas_total DOUBLE,
  n_caldeiras_carga BIGINT,
  periodo STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.silver_forecast
-- Linhas: 30 | Previsao ensemble 30 dias (10 colunas)
CREATE TABLE workspace.previsao_vapor.silver_forecast (
  data TIMESTAMP,
  RF_Horario DOUBLE,
  RF_Diario DOUBLE,
  Prophet DOUBLE,
  SARIMA DOUBLE,
  Ensemble_Media DOUBLE,
  Ensemble_Mediana DOUBLE,
  Ensemble_Ponderado DOUBLE,
  IC_lower DOUBLE,
  IC_upper DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- ============================================================
-- GOLD
-- ============================================================

-- Tabela: workspace.previsao_vapor.gold_test_modoa
-- Linhas: 3 | Resultados teste modo A (por alvo: C208, C502, C550)
CREATE TABLE workspace.previsao_vapor.gold_test_modoa (
  alvo STRING COLLATE UTF8_BINARY,
  n_treino BIGINT,
  n_teste BIGINT,
  teste_de DATE,
  teste_ate DATE,
  MAE DOUBLE,
  `WAPE_%` DOUBLE,
  R2 DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.gold_test_modob
-- Linhas: 3 | Resultados teste modo B (por horizonte: D+1, Semana, Mes)
CREATE TABLE workspace.previsao_vapor.gold_test_modob (
  alvo STRING COLLATE UTF8_BINARY,
  n_treino BIGINT,
  n_teste BIGINT,
  teste_de DATE,
  teste_ate DATE,
  MAE DOUBLE,
  `WAPE_%` DOUBLE,
  R2 DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.gold_cv_modob
-- Linhas: 12 | Cross-validation modo B (por horizonte e modelo)
CREATE TABLE workspace.previsao_vapor.gold_cv_modob (
  index BIGINT,
  horizonte STRING COLLATE UTF8_BINARY,
  modelo STRING COLLATE UTF8_BINARY,
  MAE DOUBLE,
  `WAPE_%` DOUBLE,
  R2 DOUBLE)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.gold_importancias
-- Linhas: 188 | Importancia de features (Pearson, Spearman, Gini, Permutation)
CREATE TABLE workspace.previsao_vapor.gold_importancias (
  feature STRING COLLATE UTF8_BINARY,
  categoria STRING COLLATE UTF8_BINARY,
  pearson_treino DOUBLE,
  spearman_treino DOUBLE,
  gini DOUBLE,
  perm_MAE DOUBLE,
  modo STRING COLLATE UTF8_BINARY,
  alvo STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- Tabela: workspace.previsao_vapor.gold_results_summary
-- Linhas: 5 | Resumo dos modelos treinados
CREATE TABLE workspace.previsao_vapor.gold_results_summary (
  modelo STRING COLLATE UTF8_BINARY,
  rmse_train DOUBLE,
  rmse_test DOUBLE,
  mae_test DOUBLE,
  r2_test DOUBLE,
  mape_test DOUBLE,
  n_train DOUBLE,
  n_test DOUBLE,
  best_params STRING COLLATE UTF8_BINARY)
USING delta
TBLPROPERTIES (
  'delta.enableDeletionVectors' = 'true',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7',
  'delta.parquet.compression.codec' = 'zstd');

-- ============================================================
-- UC VOLUME (modelos binarios .joblib)
-- ============================================================
-- Criado via: CREATE VOLUME IF NOT EXISTS workspace.previsao_vapor.models
-- Arquivos: /Volumes/workspace/previsao_vapor/models/*.joblib
