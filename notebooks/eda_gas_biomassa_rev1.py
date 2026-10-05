# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # EDA — Caldeiras a gás × Biomassa (novo escopo do TCC)
# MAGIC
# MAGIC **Objetivo desta etapa:** entender os dados antes de modelar.
# MAGIC
# MAGIC 1. Cobertura de dados por tag (lacunas, principalmente da biomassa `21FI551-10`)
# MAGIC 2. Estados de operação de cada caldeira a gás (desligada / standby / carga)
# MAGIC 3. **Demanda total de vapor** = vapor das 3 caldeiras a gás + biomassa (balanço de massa)
# MAGIC 4. **Consumo específico** (gás por unidade de vapor) de cada caldeira no período **antes** da biomassa
# MAGIC 5. **Gás evitado** após a partida da biomassa (estimativa contrafactual)
# MAGIC 6. Regime novo: standby e **eventos de pico** das caldeiras a gás
# MAGIC 7. Tabela **diária** salva no `2_silver` para a etapa de previsão
# MAGIC
# MAGIC Os limiares de estado (célula de configuração) são parâmetros iniciais derivados dos próprios dados —
# MAGIC **ajuste com o conhecimento da operação** depois de ver os histogramas.

# COMMAND ----------

# DBTITLE 1,Imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# COMMAND ----------

# DBTITLE 1,Configuração: tags, caminhos e limiares
# Mapeamento informado pela operação
CALDEIRAS = {
    'C208': {'vapor': '11FI208-02', 'gas': '11FC208-106', 'ar': '11FC208-138'},
    'C502': {'vapor': '11FI502-30', 'gas': '11FC502-01',  'ar': '11FC502-04'},
    'C550': {'vapor': '21FI550-30', 'gas': '21FC550-72',  'ar': '21FC550-69'},
}
BIOMASSA = {'vapor': '21FI551-10'}

# Limiares de estado, como FRAÇÃO do percentil 95 de cada tag (ajustáveis)
FRAC_VAPOR_DESLIGADA = 0.02   # vapor < 2% do p95  -> sem geração
FRAC_VAPOR_CARGA     = 0.15   # vapor >= 15% do p95 -> caldeira em carga
FRAC_GAS_STANDBY     = 0.01   # gás >= 1% do p95 sem vapor relevante -> standby (queima de manutenção)
FRAC_BIOMASSA_ON     = 0.05   # biomassa considerada em operação acima de 5% do seu p95

MIN_HORAS_DIA = 20            # dia só é válido com >= 20 h de dados

# COMMAND ----------

# DBTITLE 1,Resolver nomes das tags no bronze_processo (Delta table)
# ============================================================
# Resolver nomes das tags no Delta table (Unity Catalog)
# ============================================================


def norm(tag):
    return tag.upper().replace('.PV', '').strip()


def chave_malha(tag):
    """Área + malha sem o tipo de instrumento: '11FC502-30' -> ('11', '502-30')."""
    m = re.match(r'^(\d+)[A-Z]+(\d+-\d+)', norm(tag))
    return m.groups() if m else None


# Listar tags do Delta table
tags_disponiveis = sorted([r["tag"] for r in spark.table("workspace.previsao_vapor.bronze_processo").select("tag").distinct().collect()])
por_norm = {}
for t in tags_disponiveis:
    por_norm.setdefault(norm(t), t)


def resolver(tag_pedida):
    """Procura a tag pelo nome; se não achar, tenta pela área+malha (FI x FC)."""
    if norm(tag_pedida) in por_norm:
        return por_norm[norm(tag_pedida)], 'exato'
    k = chave_malha(tag_pedida)
    candidatos = [t for t in tags_disponiveis if k and chave_malha(t) == k and '_TOT' not in t.upper()]
    if len(candidatos) == 1:
        return candidatos[0], 'por malha'
    return None, f'não encontrada (candidatos: {candidatos})'


MAPA = {}   # nome_logico -> tag real no bronze
linhas = []
for cald, d in {**CALDEIRAS, 'BIO': BIOMASSA}.items():
    for var, tag in d.items():
        real, modo = resolver(tag)
        nome = f'{cald}_{var}'
        linhas.append({'variavel': nome, 'tag_informada': tag, 'tag_no_bronze': real, 'resolucao': modo})
        if real:
            MAPA[nome] = real

df_mapa = pd.DataFrame(linhas)
print(f"Tags distintas no bronze_processo: {len(tags_disponiveis)}")
display(df_mapa)

faltando = df_mapa[df_mapa['tag_no_bronze'].isna()]
if len(faltando):
    print("\n>>> ATENÇÃO: tags não encontradas no bronze_processo — precisam ser extraídas do IP21:")
    for _, r in faltando.iterrows():
        print(f"    {r['variavel']}: {r['tag_informada']}")
    print("\nTags do bronze com área 208/502/550/551 (para conferência):")
    print([t for t in tags_disponiveis if re.search(r'(208|502|550|551)', t)])

# COMMAND ----------

# DBTITLE 1,Carregar séries do Delta table e reamostrar para 1 h
# ============================================================
# Carregar séries do Delta table e reamostrar para 1 h
# ============================================================
from pyspark.sql.functions import col, date_trunc, avg

needed_tags = list(set(MAPA.values()))
bronze_spark = spark.table("workspace.previsao_vapor.bronze_processo").filter(col("tag").isin(needed_tags))
hourly = (bronze_spark
    .withColumn("hour", date_trunc("hour", col("ts")))
    .groupBy("hour", "tag")
    .agg(avg("value").alias("value")))
hourly_wide = hourly.groupBy("hour").pivot("tag").agg(avg("value"))

h = hourly_wide.toPandas()
h = h.set_index(pd.to_datetime(h["hour"])).drop(columns=["hour"])
h = h.sort_index().asfreq('h')

# Renomear colunas: tag real -> nome lógico (MAPA)
rename_map = {real: nome for nome, real in MAPA.items() if real in h.columns}
h = h.rename(columns=rename_map)

print(f"Tabela horária: {h.shape} | {h.index.min()} a {h.index.max()}")
display(h.describe().T.reset_index().rename(columns={'index': 'variavel'}).round(2))

# COMMAND ----------

# DBTITLE 1,Cobertura de dados por mês (% de horas com valor)
cobertura = h.notna().groupby(h.index.to_period('M')).mean().T * 100
cobertura.columns = cobertura.columns.astype(str)

fig, ax = plt.subplots(figsize=(max(12, 0.45 * cobertura.shape[1]), 0.5 * len(cobertura) + 2))
sns.heatmap(cobertura, cmap='RdYlGn', vmin=0, vmax=100, annot=cobertura.shape[1] <= 36,
            fmt='.0f', cbar_kws={'label': '% de horas com dado'}, ax=ax)
ax.set_title('Cobertura mensal de dados por variável')
plt.tight_layout()
plt.show()

if 'BIO_vapor' in h:
    b = h['BIO_vapor']
    print(f"Biomassa (21FI551-10): {b.notna().sum():,} horas com valor | "
          f"primeira: {b.first_valid_index()} | última: {b.last_valid_index()}")

# COMMAND ----------

# DBTITLE 1,Partida da biomassa (detectada nos dados)
if 'BIO_vapor' in h and h['BIO_vapor'].notna().any():
    b = h['BIO_vapor']
    lim_bio = FRAC_BIOMASSA_ON * b.quantile(0.95)
    bio_dia = b.resample('D').mean()
    INICIO_BIOMASSA = bio_dia[bio_dia > lim_bio].index.min()
else:
    INICIO_BIOMASSA = None

print(f"Início da biomassa detectado: {INICIO_BIOMASSA}  (operação informou: maio)")
print("Se a data detectada não bater com a partida real, fixe INICIO_BIOMASSA manualmente abaixo.")
# INICIO_BIOMASSA = pd.Timestamp('2026-05-01')

# COMMAND ----------

# DBTITLE 1,Estados de operação das caldeiras a gás
def estados(cald):
    v, g = h.get(f'{cald}_vapor'), h.get(f'{cald}_gas')
    if v is None:
        return None
    lim_off, lim_carga = FRAC_VAPOR_DESLIGADA * v.quantile(0.95), FRAC_VAPOR_CARGA * v.quantile(0.95)
    e = pd.Series('sem_dado', index=h.index)
    e[v.notna() & (v < lim_off)] = 'desligada'
    if g is not None:
        lim_gas = FRAC_GAS_STANDBY * g.quantile(0.95)
        e[v.notna() & (v < lim_carga) & (g >= lim_gas)] = 'standby'
    e[v.notna() & (v >= lim_carga)] = 'carga'
    e[v.notna() & (e == 'sem_dado')] = 'baixa_carga'
    return e


EST = pd.DataFrame({c: estados(c) for c in CALDEIRAS if f'{c}_vapor' in h})

# Histogramas para calibrar os limiares
fig, axes = plt.subplots(2, len(EST.columns), figsize=(6 * len(EST.columns), 8))
axes = np.array(axes).reshape(2, -1)
for j, c in enumerate(EST.columns):
    for i, var in enumerate(['vapor', 'gas']):
        s = h.get(f'{c}_{var}')
        ax = axes[i, j]
        if s is None:
            ax.set_visible(False)
            continue
        ax.hist(s.dropna(), bins=100)
        ax.set_yscale('log')
        ax.set_title(f'{c} — {var}')
        frac = FRAC_VAPOR_CARGA if var == 'vapor' else FRAC_GAS_STANDBY
        ax.axvline(frac * s.quantile(0.95), color='red', ls='--', label='limiar')
        ax.legend()
plt.suptitle('Distribuição das vazões (escala log) — calibre os limiares de estado', fontweight='bold')
plt.tight_layout()
plt.show()

# % de horas em cada estado, antes x depois da biomassa
periodo = pd.Series('antes', index=h.index)
if INICIO_BIOMASSA is not None:
    periodo[h.index >= INICIO_BIOMASSA] = 'depois'
tab = (EST.assign(periodo=periodo).melt(id_vars='periodo', var_name='caldeira', value_name='estado')
       .groupby(['caldeira', 'periodo'])['estado'].value_counts(normalize=True).mul(100).round(1)
       .unstack(fill_value=0).reset_index())
print("=== % das horas em cada estado ===")
display(tab)

# COMMAND ----------

# DBTITLE 1,Demanda total de vapor (balanço de massa) — base diária
vap_cols = [f'{c}_vapor' for c in CALDEIRAS if f'{c}_vapor' in h]
gas_cols = [f'{c}_gas' for c in CALDEIRAS if f'{c}_gas' in h]


def diario(s):
    """Total diário = média horária × 24, só em dias com >= MIN_HORAS_DIA horas de dado."""
    g = s.resample('D')
    tot = g.mean() * 24
    return tot.where(g.count() >= MIN_HORAS_DIA)


d = pd.DataFrame({c: diario(h[c]) for c in h.columns})

bio = d['BIO_vapor'] if 'BIO_vapor' in d else pd.Series(np.nan, index=d.index)
if INICIO_BIOMASSA is not None:
    bio = bio.where(d.index >= INICIO_BIOMASSA.normalize(), 0.0)   # antes da partida = 0
else:
    bio = bio.fillna(0.0)
d['vapor_gas_total'] = d[vap_cols].sum(axis=1, min_count=len(vap_cols))
d['vapor_biomassa'] = bio
d['demanda_vapor_total'] = d['vapor_gas_total'] + d['vapor_biomassa']
d['gas_total'] = d[gas_cols].sum(axis=1, min_count=len(gas_cols)) if gas_cols else np.nan
d['n_caldeiras_carga'] = (EST == 'carga').resample('D').sum().ge(6).sum(axis=1)  # >= 6 h em carga no dia

print(f"Dias com demanda total válida: {d['demanda_vapor_total'].notna().sum()} de {len(d)}")

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
axes[0].plot(d.index, d['demanda_vapor_total'], lw=0.8, label='Demanda total (gás + biomassa)')
axes[0].plot(d.index, d['vapor_biomassa'], lw=0.8, label='Biomassa')
if INICIO_BIOMASSA is not None:
    for ax in axes:
        ax.axvline(INICIO_BIOMASSA, color='k', ls='--', lw=1)
axes[0].set_ylabel('vapor / dia')
axes[0].legend()
axes[0].set_title('Demanda diária de vapor e participação da biomassa')

mensal = d[vap_cols + ['vapor_biomassa']].resample('MS').sum(min_count=1)
axes[1].stackplot(mensal.index, mensal.fillna(0).T.values, labels=mensal.columns, step='post')
axes[1].set_ylabel('vapor / mês')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].set_title('Geração mensal por fonte')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Consumo específico de gás por caldeira (período ANTES da biomassa, em carga)
antes = h.index < (INICIO_BIOMASSA if INICIO_BIOMASSA is not None else h.index.max() + pd.Timedelta('1h'))
esp = []
fig, axes = plt.subplots(1, len(CALDEIRAS), figsize=(6 * len(CALDEIRAS), 5))
for ax, c in zip(np.atleast_1d(axes), CALDEIRAS):
    v, g, a = h.get(f'{c}_vapor'), h.get(f'{c}_gas'), h.get(f'{c}_ar')
    if v is None or g is None:
        ax.set_visible(False)
        esp.append({'caldeira': c, 'obs': 'sem vapor ou gás no bronze'})
        continue
    m = antes & (EST[c] == 'carga') & v.notna() & g.notna()
    x, y = v[m], g[m]
    if len(x) < 50:
        esp.append({'caldeira': c, 'obs': f'poucas horas em carga ({len(x)})'})
        continue
    b1, b0 = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (b0 + b1 * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    razao = (y / x).median()
    ar_gas = (a[m] / g[m]).median() if a is not None else np.nan
    esp.append({'caldeira': c, 'horas_carga': int(m.sum()), 'gas_por_vapor_mediana': razao,
                'coef_angular_b1': b1, 'intercepto_b0': b0, 'R2': r2, 'razao_ar_gas_mediana': ar_gas})
    amostra = pd.DataFrame({'x': x, 'y': y}).sample(min(5000, len(x)), random_state=42)
    ax.scatter(amostra['x'], amostra['y'], s=4, alpha=0.3)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, b0 + b1 * xx, color='red', label=f'gás = {b0:.1f} + {b1:.3f}·vapor  (R²={r2:.3f})')
    ax.set_xlabel('vapor gerado')
    ax.set_ylabel('gás consumido')
    ax.set_title(c)
    ax.legend(fontsize=8)
plt.suptitle('Curva gás × vapor por caldeira (antes da biomassa, estado = carga)', fontweight='bold')
plt.tight_layout()
plt.show()

df_esp = pd.DataFrame(esp)
display(df_esp.round(4))

# COMMAND ----------

# DBTITLE 1,Gás evitado após a partida da biomassa (estimativa contrafactual)
# Hipótese: sem a biomassa, o vapor que ela gerou teria vindo das caldeiras a gás.
# Estimativa A: vapor da biomassa × consumo específico mediano (ponderado pelas horas em carga)
# Estimativa B: vapor da biomassa × coeficiente angular médio (custo marginal; o intercepto
#               já é pago pelas caldeiras que ficam em standby)
ok = df_esp.dropna(subset=['gas_por_vapor_mediana']) if 'gas_por_vapor_mediana' in df_esp else pd.DataFrame()
if INICIO_BIOMASSA is not None and len(ok):
    w = ok['horas_carga'] / ok['horas_carga'].sum()
    esp_medio = (ok['gas_por_vapor_mediana'] * w).sum()
    b1_medio = (ok['coef_angular_b1'] * w).sum()
    pos = d.loc[d.index >= INICIO_BIOMASSA.normalize()]
    evit = pd.DataFrame({
        'vapor_biomassa': pos['vapor_biomassa'],
        'gas_evitado_A': pos['vapor_biomassa'] * esp_medio,
        'gas_evitado_B': pos['vapor_biomassa'] * b1_medio,
        'gas_consumido_real': pos['gas_total'],
    }).resample('MS').sum(min_count=1)
    evit['dias_validos'] = pos['vapor_biomassa'].notna().resample('MS').sum()
    print(f"Consumo específico médio ponderado = {esp_medio:.4f} | coef. angular médio = {b1_medio:.4f}")
    display(evit.reset_index().round(1))
    print("ATENÇÃO: meses com poucos dias válidos (lacunas do 21FI551-10) SUBESTIMAM o gás evitado.")
else:
    print("Sem data de partida da biomassa ou sem curva gás × vapor — estimativa não calculada.")

# COMMAND ----------

# DBTITLE 1,Regime novo: standby e eventos de pico das caldeiras a gás
if INICIO_BIOMASSA is not None:
    pos_h = h.index >= INICIO_BIOMASSA
    eventos = []
    for c in EST.columns:
        em_carga = (EST[c] == 'carga') & pos_h
        grupo = (em_carga != em_carga.shift()).cumsum()
        for _, blk in em_carga[em_carga].groupby(grupo[em_carga]):
            ini, fim = blk.index.min(), blk.index.max()
            jan = slice(ini, fim)
            eventos.append({
                'caldeira': c, 'inicio': ini, 'duracao_h': len(blk),
                'gas_consumido': h.get(f'{c}_gas', pd.Series(dtype=float))[jan].sum(),
                'biomassa_media_no_evento': h['BIO_vapor'][jan].mean() if 'BIO_vapor' in h else np.nan,
            })
    df_ev = pd.DataFrame(eventos)
    bio_mediana = h.loc[pos_h, 'BIO_vapor'].median() if 'BIO_vapor' in h else np.nan
    print(f"Eventos de carga das caldeiras a gás após a biomassa: {len(df_ev)}")
    if len(df_ev):
        df_ev['biomassa_abaixo_da_mediana'] = df_ev['biomassa_media_no_evento'] < bio_mediana
        display(df_ev.groupby('caldeira').agg(
            n_eventos=('inicio', 'size'), duracao_media_h=('duracao_h', 'mean'),
            duracao_max_h=('duracao_h', 'max'), gas_total=('gas_consumido', 'sum'),
            pct_com_biomassa_baixa=('biomassa_abaixo_da_mediana', 'mean')).reset_index().round(2))
        display(df_ev.sort_values('inicio').reset_index(drop=True))

    # Consumo de gás em standby (por hora) após a biomassa
    sb = []
    for c in EST.columns:
        g = h.get(f'{c}_gas')
        if g is None:
            continue
        m = pos_h & (EST[c] == 'standby')
        sb.append({'caldeira': c, 'horas_standby': int(m.sum()), 'gas_medio_h_standby': g[m].mean(),
                   'gas_total_standby': g[m].sum()})
    print("\n=== Standby após a biomassa ===")
    display(pd.DataFrame(sb).round(3))

# COMMAND ----------

# DBTITLE 1,Salvar tabela diária no 2_silver (Delta table)
# ============================================================
# Salvar tabela diária no Delta table (Unity Catalog)
# ============================================================
d_out = d.copy()
d_out['periodo'] = np.where(
    (INICIO_BIOMASSA is not None) & (d_out.index >= (INICIO_BIOMASSA.normalize() if INICIO_BIOMASSA is not None else d_out.index.max())),
    'com_biomassa', 'so_gas')

spark.createDataFrame(d_out.reset_index()).write \
    .mode("overwrite").saveAsTable("workspace.previsao_vapor.silver_diario")
print(f"Salvo: workspace.previsao_vapor.silver_diario | {d_out.shape}")
display(d_out.reset_index().tail(10))