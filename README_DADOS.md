# Dados no Unity Catalog — workspace.previsao_vapor

## Visão Geral

Os dados do projeto **Previsão de Consumo de Vapor** estão armazenados no Unity Catalog
no schema `workspace.previsao_vapor`. O repositório Git contém apenas código
(notebooks) e a documentação do schema (`schema.sql`); os dados ficam no UC.

## Arquitetura

```
Git (código)                    Unity Catalog (dados)
├── notebooks/                  ├── bronze_clima
├── schema.sql  ────────────►    ├── bronze_processo
├── .gitignore                   ├── silver_prepared
└── README_DADOS.md              ├── silver_diario
                                 ├── silver_forecast
                                 ├── gold_test_modoA
                                 ├── gold_test_modoB
                                 ├── gold_cv_modoB
                                 ├── gold_importancias
                                 ├── gold_results_summary
                                 └── models/ (UC Volume, .joblib)
```

## Tabelas

### Bronze (dados brutos)

| Tabela | Linhas | Descrição | Colunas principais |
|---|---|---|---|
| `bronze_clima` | 48.168 | Dados meteorológicos (2021–Jun 2026) | `datetime_local`, `temperature_C`, `relative_humidity_pct`, `surface_pressure_hPa`, `pressure_msl_hPa` |
| `bronze_processo` | 308.896.910 | Dados de processo em formato long (tag, ts, value) | `tag`, `ts`, `value` |

### Silver (dados preparados)

| Tabela | Linhas | Descrição | Colunas principais |
|---|---|---|---|
| `silver_prepared` | 30.421 | Dataset preparado para Random Forest (90 colunas) | `datetime`, 4 alvos FI, totalizadores _TOT, variáveis climáticas, features temporais (hour, dayofweek, month, seno/cosseno), lags e médias móveis (1h–24h) |
| `silver_diario` | 1.361 | Consumo diário de gás e biomassa (17 colunas) | `datetime`, `C208_gas`, `C502_gas`, `C550_gas`, `BIO_vapor`, `vapor_gas_total`, `vapor_biomassa`, `demanda_vapor_total`, `periodo` |
| `silver_forecast` | 30 | Previsão ensemble 30 dias (10 colunas) | `data`, `RF_Horario`, `RF_Diario`, `Prophet`, `SARIMA`, `Ensemble_Media`, `Ensemble_Mediana`, `Ensemble_Ponderado`, `IC_lower`, `IC_upper` |

### Gold (resultados dos modelos)

| Tabela | Linhas | Descrição | Colunas principais |
|---|---|---|---|
| `gold_test_modoA` | 3 | Resultados teste modo A (por alvo: C208, C502, C550) | `alvo`, `n_treino`, `n_teste`, `teste_de`, `teste_ate`, `MAE`, `WAPE_%`, `R2` |
| `gold_test_modoB` | 3 | Resultados teste modo B (por horizonte: D+1, Semana, Mês) | `alvo`, `n_treino`, `n_teste`, `teste_de`, `teste_ate`, `MAE`, `WAPE_%`, `R2` |
| `gold_cv_modob` | 12 | Cross-validation modo B (por horizonte e modelo) | `horizonte`, `modelo`, `MAE`, `WAPE_%`, `R2` |
| `gold_importancias` | 188 | Importância de features (Pearson, Spearman, Gini, Permutation) | `feature`, `categoria`, `pearson_treino`, `spearman_treino`, `gini`, `perm_MAE`, `modo`, `alvo` |
| `gold_results_summary` | 5 | Resumo dos modelos treinados | `modelo`, `rmse_train`, `rmse_test`, `mae_test`, `r2_test`, `mape_test`, `n_train`, `n_test`, `best_params` |

### UC Volume (modelos binários)

| Volume | Descrição |
|---|---|
| `workspace.previsao_vapor.models` | Modelos `.joblib` do Random Forest. Acesso: `/Volumes/workspace/previsao_vapor/models/` |

## Como Recriar as Tabelas

1. Execute o notebook `migrar_para_delta` — ele cria o schema, lê os CSVs e escreve todas as tabelas.
2. Ou aplique o arquivo `schema.sql` diretamente em um SQL warehouse para recriar apenas as estruturas (sem dados).

## Como Consultar

```sql
-- Listar todas as tabelas
SHOW TABLES IN workspace.previsao_vapor;

-- Consultar tabela silver_prepared
SELECT datetime, `11FI208-02`, `11FI502-30.PV`
FROM workspace.previsao_vapor.silver_prepared
LIMIT 10;

-- Contar linhas por tabela
SELECT 'bronze_processo' AS tabela, COUNT(*) AS linhas FROM workspace.previsao_vapor.bronze_processo
UNION ALL
SELECT 'silver_prepared', COUNT(*) FROM workspace.previsao_vapor.silver_prepared;
```

## .gitignore

O arquivo `.gitignore` exclui do Git:
- `0_raw/`, `1_bronze/`, `2_silver/`, `3_gold/` — diretórios de dados
- `*.csv`, `*.joblib`, `*.pkl`, `*.parquet` — arquivos de dados e modelos
