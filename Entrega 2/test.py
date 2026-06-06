"""# Parte 2"""

from scipy.stats import t
from scipy.stats import mannwhitneyu
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, t


# =========================
# 1. Cargar archivos
# =========================

df_real = pd.read_csv("validar_supermercado.csv")
df_sim = pd.read_csv("metricas_replicas.csv")

# Quitar espacios extra en nombres de columnas
df_real.columns = df_real.columns.str.strip()
df_sim.columns = df_sim.columns.str.strip()

print("Columnas datos reales:")
print(df_real.columns.tolist())

print("\nColumnas simulación:")
print(df_sim.columns.tolist())

# Diccionario: columna simulada -> columna real
mapa_columnas_simulacion = {
    "utilidades totales": "utilidad",

    "Promedio productos almacen": "promedio_productos_Almacén",
    "Promedio productos verdureria": "promedio_productos_Verdurería",
    "Promedio productos panaderia": "promedio_productos_Panadería",
    "Promedio productos refrigerados": "promedio_productos_Refrigerados",

    "Promedio tiempo espera almacen": "tiempo_espera_Almacén",
    "Promedio tiempo espera verduleria": "tiempo_espera_Verdurería",
    "Promedio tiempo espera panaderia": "tiempo_espera_Panadería",
    "Promedio tiempo espera refrigerados": "tiempo_espera_Refrigerados",

    "Promedio tiempo espera balanza verduleria": "tiempo_balanza_Verdurería",
    "Promedio tiempo espera balanza panaderia": "tiempo_balanza_Panadería",

    "Promedio tiempo espera caja automática": "tiempo_caja_Autoservicio",
    "Promedio tiempo espera caja normal": "tiempo_caja_Normal",
    "Promedio tiempo espera caja preferencial": "tiempo_caja_Preferencial",

    "Promedio tiempo de permanencia clientes normales": "tiempo_permanencia_Normal",
    "Promedio tiempo de permanencia clientes preferenciales": "tiempo_permanencia_Preferencial",
}

df_sim_renombrado = df_sim.rename(columns=mapa_columnas_simulacion)

metricas_comunes = [
    c for c in df_real.columns
    if c in df_sim_renombrado.columns
]

print("Métricas comunes encontradas:")
for c in metricas_comunes:
    print("-", c)

print("\nCantidad de métricas comunes:", len(metricas_comunes))

df_real_val = df_real[metricas_comunes].copy()
df_sim_val = df_sim_renombrado[metricas_comunes].copy()

# Asegurar que todos los datos sean numéricos
for col in metricas_comunes:
    df_real_val[col] = pd.to_numeric(df_real_val[col], errors="coerce")
    df_sim_val[col] = pd.to_numeric(df_sim_val[col], errors="coerce")

print("Dimensión datos reales:", df_real_val.shape)
print("Dimensión datos simulados:", df_sim_val.shape)

print("\nValores perdidos datos reales:")
print(df_real_val.isna().sum())

print("\nValores perdidos datos simulados:")
print(df_sim_val.isna().sum())

"""Para el Test U de Mann-Whitney se considera:

H0: la distribución de la métrica observada y la métrica simulada no presenta diferencias estadísticamente significativas.
H1: las distribuciones son distintas.

Se usa α = 0.05. Si p-value < 0.05, se rechaza H0.
"""


alpha = 0.05
resultados_mw = []

for metrica in metricas_comunes:
    real = df_real_val[metrica].dropna()
    sim = df_sim_val[metrica].dropna()

    estadistico_u, p_value = mannwhitneyu(
        real,
        sim,
        alternative="two-sided"
    )

    resultados_mw.append({
        "metrica": metrica,
        "n_observado": len(real),
        "n_simulado": len(sim),
        "media_observado": real.mean(),
        "media_simulado": sim.mean(),
        "mediana_observado": real.median(),
        "mediana_simulado": sim.median(),
        "U_Mann_Whitney": estadistico_u,
        "p_value_Mann_Whitney": p_value,
        "acepta_H0_Mann_Whitney": p_value >= alpha
    })

resultados_mw = pd.DataFrame(resultados_mw)
resultados_mw.round(4)

"""Para el intervalo de confianza se analiza la diferencia media_simulada - media_observada.
Si el intervalo contiene 0, la diferencia entre medias es compatible con cero.
Si no contiene 0, se concluye que existe diferencia estadísticamente significativa entre ambas medias.
"""


def intervalo_confianza_diferencia_medias(observado, simulado, alpha=0.05):
    observado = np.array(observado, dtype=float)
    simulado = np.array(simulado, dtype=float)

    n_obs = len(observado)
    n_sim = len(simulado)

    media_obs = np.mean(observado)
    media_sim = np.mean(simulado)

    var_obs = np.var(observado, ddof=1)
    var_sim = np.var(simulado, ddof=1)

    diferencia = media_sim - media_obs
    error_estandar = np.sqrt(var_obs / n_obs + var_sim / n_sim)

    if error_estandar == 0:
        return diferencia, diferencia, diferencia, np.nan

    gl_num = (var_obs / n_obs + var_sim / n_sim) ** 2
    gl_den = ((var_obs / n_obs) ** 2) / (n_obs - 1) + \
        ((var_sim / n_sim) ** 2) / (n_sim - 1)
    gl = gl_num / gl_den

    t_critico = t.ppf(1 - alpha / 2, gl)

    ic_inf = diferencia - t_critico * error_estandar
    ic_sup = diferencia + t_critico * error_estandar

    return diferencia, ic_inf, ic_sup, gl


resultados_ic = []

for metrica in metricas_comunes:
    real = df_real_val[metrica].dropna()
    sim = df_sim_val[metrica].dropna()

    diferencia, ic_inf, ic_sup, gl = intervalo_confianza_diferencia_medias(
        real,
        sim,
        alpha=0.05
    )

    resultados_ic.append({
        "metrica": metrica,
        "diferencia_media_sim_menos_obs": diferencia,
        "IC_95_inf": ic_inf,
        "IC_95_sup": ic_sup,
        "gl_welch": gl,
        "IC_contiene_0": ic_inf <= 0 <= ic_sup
    })

resultados_ic = pd.DataFrame(resultados_ic)
resultados_ic.round(4)

tabla_validacion = resultados_mw.merge(
    resultados_ic,
    on="metrica",
    how="inner"
)

tabla_validacion["valida_segun_criterio"] = (
    (tabla_validacion["p_value_Mann_Whitney"] >= 0.05)
    & (tabla_validacion["IC_contiene_0"])
)

tabla_validacion = tabla_validacion.sort_values(
    by=["valida_segun_criterio", "metrica"]
)

tabla_validacion.round(4)

tabla_validacion.round(4).to_csv(
    "resultados_validacion_modelo.csv",
    index=False
)

metricas_validadas = tabla_validacion[tabla_validacion["valida_segun_criterio"]]
metricas_no_validadas = tabla_validacion[~tabla_validacion["valida_segun_criterio"]]

print("Métricas validadas:")
print(metricas_validadas["metrica"].tolist())

print("\nMétricas no validadas:")
print(metricas_no_validadas["metrica"].tolist())

n_total = len(tabla_validacion)
n_validas = tabla_validacion["valida_segun_criterio"].sum()
n_no_validas = n_total - n_validas

print(f"Métricas totales evaluadas: {n_total}")
print(f"Métricas validadas: {n_validas}")
print(f"Métricas no validadas: {n_no_validas}")
print(f"Porcentaje de métricas validadas: {100 * n_validas / n_total:.2f}%")

# ========================================================
# 2. Validación de Métricas Extra
# ========================================================

print("\n" + "="*50)
print("Validación de Métricas Extra")
print("="*50)

df_extra = pd.read_csv("metricas_extra.csv")
df_sim_extra = pd.read_csv("metricas_replicas.csv")

df_extra.columns = df_extra.columns.str.strip()
df_sim_extra.columns = df_sim_extra.columns.str.strip()

mapa_columnas_simulacion_extra = {
    "utilidad almacen": "utilidad_Almacén",
    "utilidad verduleria": "utilidad_Verdurería",
    "utilidad panaderia": "utilidad_Panadería",
    "utilidad refrigerados": "utilidad_Refrigerados",
    "cantidad clientes visita almacen": "clientes_Almacén",
    "cantidad clientes visita verduleria": "clientes_Verdurería",
    "cantidad clientes visita panaderia": "clientes_Panadería",
    "cantidad clientes visita refrigerados": "clientes_Refrigerados",
    "cantidad clientes caja automática": "clientes_caja_Autoservicio",
    "cantidad clientes caja normal": "clientes_caja_Normal",
    "cantidad clientes caja preferencial": "clientes_caja_Preferencial",
    "cantidad clientes rechazados por capacidad": "clientes_rechazados"
}

df_sim_extra_renombrado = df_sim_extra.rename(columns=mapa_columnas_simulacion_extra)

metricas_comunes_extra = [
    c for c in df_extra.columns
    if c in df_sim_extra_renombrado.columns
]

print("Métricas extra encontradas:")
for c in metricas_comunes_extra:
    print("-", c)

df_extra_val = df_extra[metricas_comunes_extra].copy()
df_sim_extra_val = df_sim_extra_renombrado[metricas_comunes_extra].copy()

# Asegurar numérico
for col in metricas_comunes_extra:
    df_extra_val[col] = pd.to_numeric(df_extra_val[col], errors="coerce")
    df_sim_extra_val[col] = pd.to_numeric(df_sim_extra_val[col], errors="coerce")

resultados_mw_extra = []

for metrica in metricas_comunes_extra:
    real = df_extra_val[metrica].dropna()
    sim = df_sim_extra_val[metrica].dropna()

    if len(real) == 0 or len(sim) == 0:
        continue

    try:
        estadistico_u, p_value = mannwhitneyu(
            real,
            sim,
            alternative="two-sided"
        )
    except ValueError:
        estadistico_u = np.nan
        p_value = np.nan

    resultados_mw_extra.append({
        "metrica": metrica,
        "n_observado": len(real),
        "n_simulado": len(sim),
        "media_observado": real.mean(),
        "media_simulado": sim.mean(),
        "mediana_observado": real.median(),
        "mediana_simulado": sim.median(),
        "U_Mann_Whitney": estadistico_u,
        "p_value_Mann_Whitney": p_value,
        "acepta_H0_Mann_Whitney": p_value >= alpha if pd.notna(p_value) else True
    })

resultados_mw_extra = pd.DataFrame(resultados_mw_extra)

resultados_ic_extra = []

for metrica in metricas_comunes_extra:
    real = df_extra_val[metrica].dropna()
    sim = df_sim_extra_val[metrica].dropna()

    if len(real) == 0 or len(sim) == 0:
        continue

    diferencia, ic_inf, ic_sup, gl = intervalo_confianza_diferencia_medias(
        real,
        sim,
        alpha=0.05
    )

    resultados_ic_extra.append({
        "metrica": metrica,
        "diferencia_media_sim_menos_obs": diferencia,
        "IC_95_inf": ic_inf,
        "IC_95_sup": ic_sup,
        "gl_welch": gl,
        "IC_contiene_0": (ic_inf <= 0 <= ic_sup) if pd.notna(ic_inf) else True
    })

resultados_ic_extra = pd.DataFrame(resultados_ic_extra)

if not resultados_mw_extra.empty and not resultados_ic_extra.empty:
    tabla_validacion_extra = resultados_mw_extra.merge(
        resultados_ic_extra,
        on="metrica",
        how="inner"
    )

    tabla_validacion_extra["valida_segun_criterio"] = (
        (tabla_validacion_extra["p_value_Mann_Whitney"] >= 0.05)
        & (tabla_validacion_extra["IC_contiene_0"])
    )

    tabla_validacion_extra = tabla_validacion_extra.sort_values(
        by=["valida_segun_criterio", "metrica"]
    )

    tabla_validacion_extra.round(4).to_csv(
        "resultados_validacion_modelo_extra.csv",
        index=False
    )

    metricas_validadas_extra = tabla_validacion_extra[tabla_validacion_extra["valida_segun_criterio"]]
    metricas_no_validadas_extra = tabla_validacion_extra[~tabla_validacion_extra["valida_segun_criterio"]]

    print("\nMétricas extra validadas:")
    print(metricas_validadas_extra["metrica"].tolist())

    print("\nMétricas extra no validadas:")
    print(metricas_no_validadas_extra["metrica"].tolist())

    n_total_extra = len(tabla_validacion_extra)
    n_validas_extra = tabla_validacion_extra["valida_segun_criterio"].sum()
    n_no_validas_extra = n_total_extra - n_validas_extra

    print(f"\nMétricas totales extra evaluadas: {n_total_extra}")
    print(f"Métricas extra validadas: {n_validas_extra}")
    print(f"Métricas extra no validadas: {n_no_validas_extra}")
    if n_total_extra > 0:
        print(f"Porcentaje de métricas extra validadas: {100 * n_validas_extra / n_total_extra:.2f}%")
else:
    print("\nNo se encontraron métricas extra comunes o no hay datos para evaluarlas.")

# ========================================================
# 3. Guardar Reporte en Archivo de Texto
# ========================================================

def generar_reporte_txt(nombre_archivo="resumen_validacion.txt"):
    try:
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE VALIDACIÓN DEL MODELO DE SIMULACIÓN\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- PARTE 1: VALIDACIÓN BASE ---\n")
            f.write(f"Métricas totales evaluadas: {n_total}\n")
            f.write(f"Métricas validadas: {n_validas}\n")
            f.write(f"Métricas no validadas: {n_no_validas}\n")
            if n_total > 0:
                f.write(f"Porcentaje de métricas validadas: {100 * n_validas / n_total:.2f}%\n")
            
            f.write("\nLista de Métricas Validadas:\n")
            for _, row in metricas_validadas.iterrows():
                f.write(f"  - {row['metrica']} (p-value MW: {row['p_value_Mann_Whitney']:.4f}, IC 95%: [{row['IC_95_inf']:.4f}, {row['IC_95_sup']:.4f}])\n")
                
            f.write("\nLista de Métricas No Validadas:\n")
            for _, row in metricas_no_validadas.iterrows():
                f.write(f"  - {row['metrica']} (p-value MW: {row['p_value_Mann_Whitney']:.4f}, IC 95%: [{row['IC_95_inf']:.4f}, {row['IC_95_sup']:.4f}])\n")

            f.write("\n" + "-" * 60 + "\n\n")
            
            f.write("--- PARTE 2: VALIDACIÓN MÉTRICAS EXTRA ---\n")
            if 'n_total_extra' in globals():
                f.write(f"Métricas totales extra evaluadas: {n_total_extra}\n")
                f.write(f"Métricas extra validadas: {n_validas_extra}\n")
                f.write(f"Métricas extra no validadas: {n_no_validas_extra}\n")
                if n_total_extra > 0:
                    f.write(f"Porcentaje de métricas extra validadas: {100 * n_validas_extra / n_total_extra:.2f}%\n")
                
                f.write("\nLista de Métricas Extra Validadas:\n")
                for _, row in metricas_validadas_extra.iterrows():
                    f.write(f"  - {row['metrica']} (p-value MW: {row['p_value_Mann_Whitney']:.4f}, IC 95%: [{row['IC_95_inf']:.4f}, {row['IC_95_sup']:.4f}])\n")
                    
                f.write("\nLista de Métricas Extra No Validadas:\n")
                for _, row in metricas_no_validadas_extra.iterrows():
                    f.write(f"  - {row['metrica']} (p-value MW: {row['p_value_Mann_Whitney']:.4f}, IC 95%: [{row['IC_95_inf']:.4f}, {row['IC_95_sup']:.4f}])\n")
            else:
                f.write("No se evaluaron métricas extra.\n")
                
            f.write("\n" + "=" * 60 + "\n")
        print(f"\nSe ha guardado un reporte ordenado en: {nombre_archivo}")
    except Exception as e:
        print(f"\nError al guardar el reporte: {e}")

generar_reporte_txt()
