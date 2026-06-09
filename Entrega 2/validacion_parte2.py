"""
Validación Parte 2 - Proyecto de Simulación de Supermercado

Este script ejecuta la simulación del repositorio actual, carga las métricas simuladas
(metricas_replicas.csv) y las compara con los datos observados de validación.

Archivo obligatorio externo:
    - validar_supermercado.csv

Archivo opcional externo:
    - metricas_extra.csv

Uso básico, desde la carpeta donde están main.py, simulacion.py, clases.py y parametros.py:
    python validacion_parte2.py

Uso si no quieres volver a correr la simulación y ya existe metricas_replicas.csv:
    python validacion_parte2.py --skip-run

Uso indicando rutas explícitas:
    python validacion_parte2.py --observado validar_supermercado.csv --extra metricas_extra.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t


# =========================================================
# Configuración de columnas
# =========================================================

MAPA_COLUMNAS_BASE: Dict[str, str] = {
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

MAPA_COLUMNAS_EXTRA: Dict[str, str] = {
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
    "cantidad clientes rechazados por capacidad": "clientes_rechazados",
    "Cantidad total de clientes atendidos": "clientes_atendidos",
}

ARCHIVOS_CODIGO_REQUERIDOS = ["main.py", "simulacion.py", "clases.py", "parametros.py"]


# =========================================================
# Utilidades generales
# =========================================================

def limpiar_nombre_archivo(texto: str) -> str:
    reemplazos = {
        " ": "_", "/": "_", "\\": "_", ":": "_",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
    }
    salida = texto
    for original, nuevo in reemplazos.items():
        salida = salida.replace(original, nuevo)
    return salida


def verificar_archivos_codigo(base_dir: Path) -> None:
    faltantes = [nombre for nombre in ARCHIVOS_CODIGO_REQUERIDOS if not (base_dir / nombre).exists()]
    if faltantes:
        raise FileNotFoundError(
            "No se encontraron estos archivos de código en la carpeta actual: "
            + ", ".join(faltantes)
            + "\nUbica este script en la misma carpeta que main.py, simulacion.py, clases.py y parametros.py."
        )


def leer_csv(path: Path, descripcion: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {descripcion}: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def ejecutar_simulacion(base_dir: Path, archivo_simulado: Path, skip_run: bool) -> None:
    if skip_run and archivo_simulado.exists():
        print(f"Se omite la simulación porque ya existe {archivo_simulado.name}.")
        return

    print("Ejecutando main.py para generar metricas_replicas.csv...")
    subprocess.run([sys.executable, "main.py"], cwd=base_dir, check=True)

    if not archivo_simulado.exists():
        raise FileNotFoundError(
            "Se ejecutó main.py, pero no se generó metricas_replicas.csv. "
            "Revisa que main.py esté guardando ese archivo."
        )


def preparar_bases_comparables(
    df_observado: pd.DataFrame,
    df_simulado: pd.DataFrame,
    mapa_columnas: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Renombra columnas simuladas para que calcen con las observadas y deja solo métricas comunes."""

    df_sim_renombrado = df_simulado.rename(columns=mapa_columnas)

    metricas_comunes = [col for col in df_observado.columns if col in df_sim_renombrado.columns]

    df_obs_val = df_observado[metricas_comunes].copy()
    df_sim_val = df_sim_renombrado[metricas_comunes].copy()

    for col in metricas_comunes:
        df_obs_val[col] = pd.to_numeric(df_obs_val[col], errors="coerce")
        df_sim_val[col] = pd.to_numeric(df_sim_val[col], errors="coerce")

    return df_obs_val, df_sim_val, metricas_comunes


# =========================================================
# Métodos estadísticos
# =========================================================

def intervalo_confianza_diferencia_medias(
    observado: Iterable[float],
    simulado: Iterable[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """IC de Welch para diferencia de medias: media_simulada - media_observada."""

    observado_arr = np.array(list(observado), dtype=float)
    simulado_arr = np.array(list(simulado), dtype=float)

    n_obs = len(observado_arr)
    n_sim = len(simulado_arr)

    media_obs = np.mean(observado_arr)
    media_sim = np.mean(simulado_arr)
    var_obs = np.var(observado_arr, ddof=1)
    var_sim = np.var(simulado_arr, ddof=1)

    diferencia = media_sim - media_obs
    error_estandar = np.sqrt(var_obs / n_obs + var_sim / n_sim)

    if error_estandar == 0:
        return diferencia, diferencia, diferencia, np.nan

    gl_num = (var_obs / n_obs + var_sim / n_sim) ** 2
    gl_den = ((var_obs / n_obs) ** 2) / (n_obs - 1) + ((var_sim / n_sim) ** 2) / (n_sim - 1)
    gl = gl_num / gl_den

    t_critico = t.ppf(1 - alpha / 2, gl)
    ic_inf = diferencia - t_critico * error_estandar
    ic_sup = diferencia + t_critico * error_estandar

    return diferencia, ic_inf, ic_sup, gl


def validar_metricas(
    df_obs_val: pd.DataFrame,
    df_sim_val: pd.DataFrame,
    metricas: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    resultados = []

    for metrica in metricas:
        observado = df_obs_val[metrica].dropna()
        simulado = df_sim_val[metrica].dropna()

        if len(observado) < 2 or len(simulado) < 2:
            resultados.append({
                "metrica": metrica,
                "n_observado": len(observado),
                "n_simulado": len(simulado),
                "media_observado": observado.mean() if len(observado) else np.nan,
                "media_simulado": simulado.mean() if len(simulado) else np.nan,
                "mediana_observado": observado.median() if len(observado) else np.nan,
                "mediana_simulado": simulado.median() if len(simulado) else np.nan,
                "U_Mann_Whitney": np.nan,
                "p_value_Mann_Whitney": np.nan,
                "acepta_H0_Mann_Whitney": False,
                "diferencia_media_sim_menos_obs": np.nan,
                "IC_95_inf": np.nan,
                "IC_95_sup": np.nan,
                "gl_welch": np.nan,
                "IC_contiene_0": False,
                "valida_segun_criterio": False,
                "comentario": "No hay suficientes datos para comparar.",
            })
            continue

        try:
            estadistico_u, p_value = mannwhitneyu(observado, simulado, alternative="two-sided")
        except ValueError:
            estadistico_u, p_value = np.nan, np.nan

        diferencia, ic_inf, ic_sup, gl = intervalo_confianza_diferencia_medias(
            observado,
            simulado,
            alpha=alpha,
        )

        acepta_mw = bool(pd.notna(p_value) and p_value >= alpha)
        ic_contiene_0 = bool(pd.notna(ic_inf) and pd.notna(ic_sup) and ic_inf <= 0 <= ic_sup)
        valida = acepta_mw and ic_contiene_0

        resultados.append({
            "metrica": metrica,
            "n_observado": len(observado),
            "n_simulado": len(simulado),
            "media_observado": observado.mean(),
            "media_simulado": simulado.mean(),
            "mediana_observado": observado.median(),
            "mediana_simulado": simulado.median(),
            "U_Mann_Whitney": estadistico_u,
            "p_value_Mann_Whitney": p_value,
            "acepta_H0_Mann_Whitney": acepta_mw,
            "diferencia_media_sim_menos_obs": diferencia,
            "IC_95_inf": ic_inf,
            "IC_95_sup": ic_sup,
            "gl_welch": gl,
            "IC_contiene_0": ic_contiene_0,
            "valida_segun_criterio": valida,
            "comentario": "Validada" if valida else "No validada",
        })

    tabla = pd.DataFrame(resultados)
    if not tabla.empty:
        tabla = tabla.sort_values(by=["valida_segun_criterio", "metrica"]).reset_index(drop=True)
    return tabla


# =========================================================
# Gráficos y reportes
# =========================================================

def crear_histogramas(
    df_obs_val: pd.DataFrame,
    df_sim_val: pd.DataFrame,
    metricas: List[str],
    carpeta_salida: Path,
) -> None:
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    for metrica in metricas:
        observado = df_obs_val[metrica].dropna()
        simulado = df_sim_val[metrica].dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(observado, bins=20, alpha=0.6, label="Datos observados", density=True)
        plt.hist(simulado, bins=20, alpha=0.6, label="Simulación", density=True)
        plt.title(f"Histograma comparativo: {metrica}")
        plt.xlabel(metrica)
        plt.ylabel("Densidad")
        plt.legend()
        plt.tight_layout()

        nombre_archivo = limpiar_nombre_archivo(metrica) + ".png"
        plt.savefig(carpeta_salida / nombre_archivo, dpi=150)
        plt.close()


def escribir_resumen(
    path: Path,
    tabla_base: pd.DataFrame,
    tabla_extra: Optional[pd.DataFrame],
    archivo_observado: Path,
    archivo_extra: Optional[Path],
) -> None:
    def escribir_bloque(f, titulo: str, tabla: pd.DataFrame) -> None:
        f.write(f"{titulo}\n")
        f.write("-" * 60 + "\n")

        if tabla.empty:
            f.write("No se evaluaron métricas.\n\n")
            return

        n_total = len(tabla)
        n_validadas = int(tabla["valida_segun_criterio"].sum())
        n_no_validadas = n_total - n_validadas
        porcentaje = 100 * n_validadas / n_total if n_total > 0 else 0

        f.write(f"Métricas evaluadas: {n_total}\n")
        f.write(f"Métricas validadas: {n_validadas}\n")
        f.write(f"Métricas no validadas: {n_no_validadas}\n")
        f.write(f"Porcentaje de métricas validadas: {porcentaje:.2f}%\n\n")

        f.write("Métricas validadas:\n")
        for metrica in tabla.loc[tabla["valida_segun_criterio"], "metrica"]:
            f.write(f"- {metrica}\n")

        f.write("\nMétricas no validadas:\n")
        for metrica in tabla.loc[~tabla["valida_segun_criterio"], "metrica"]:
            f.write(f"- {metrica}\n")
        f.write("\n")

    with path.open("w", encoding="utf-8") as f:
        f.write("REPORTE DE VALIDACIÓN DEL MODELO\n")
        f.write("=" * 60 + "\n\n")
        f.write("ARCHIVOS USADOS\n")
        f.write(f"- Código del repositorio actual\n")
        f.write(f"- Métricas simuladas: metricas_replicas.csv\n")
        f.write(f"- Datos observados base: {archivo_observado.name}\n")
        if archivo_extra is not None:
            f.write(f"- Datos observados extra: {archivo_extra.name}\n")
        else:
            f.write("- Datos observados extra: no se entregó archivo extra\n")
        f.write("\n")

        f.write("CRITERIO DE VALIDACIÓN\n")
        f.write("- Nivel de significancia: alpha = 0.05\n")
        f.write("- Método 1: Test U de Mann-Whitney para comparar distribuciones.\n")
        f.write("- Método 2: IC 95% de Welch para la diferencia de medias simulada - observada.\n")
        f.write("- Una métrica se considera validada si p-value >= 0.05 y el IC 95% contiene 0.\n\n")

        escribir_bloque(f, "VALIDACIÓN BASE", tabla_base)
        if tabla_extra is not None:
            escribir_bloque(f, "VALIDACIÓN EXTRA", tabla_extra)


def comprimir_resultados(carpeta_resultados: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for archivo in carpeta_resultados.rglob("*"):
            if archivo.is_file():
                zf.write(archivo, arcname=archivo.relative_to(carpeta_resultados.parent))


def imprimir_resumen_consola(nombre: str, tabla: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(nombre)
    print("=" * 60)

    if tabla.empty:
        print("No se evaluaron métricas.")
        return

    n_total = len(tabla)
    n_validadas = int(tabla["valida_segun_criterio"].sum())
    n_no_validadas = n_total - n_validadas
    print(f"Métricas evaluadas: {n_total}")
    print(f"Métricas validadas: {n_validadas}")
    print(f"Métricas no validadas: {n_no_validadas}")
    print(f"Porcentaje validado: {100 * n_validadas / n_total:.2f}%")

    print("\nNo validadas:")
    print(tabla.loc[~tabla["valida_segun_criterio"], "metrica"].tolist())


# =========================================================
# Flujo principal
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validación Parte 2 del proyecto de simulación.")
    parser.add_argument("--observado", default="validar_supermercado.csv", help="CSV observado base de la profesora.")
    parser.add_argument("--extra", default="metricas_extra.csv", help="CSV observado extra opcional.")
    parser.add_argument("--simulado", default="metricas_replicas.csv", help="CSV de métricas simuladas generado por main.py.")
    parser.add_argument("--out", default="resultados_parte_2_validacion", help="Carpeta de salida.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia.")
    parser.add_argument("--skip-run", action="store_true", help="No ejecuta main.py si ya existe metricas_replicas.csv.")
    parser.add_argument("--no-zip", action="store_true", help="No genera archivo zip con los resultados.")
    args = parser.parse_args()

    base_dir = Path.cwd()
    verificar_archivos_codigo(base_dir)

    archivo_observado = base_dir / args.observado
    archivo_extra = base_dir / args.extra
    archivo_simulado = base_dir / args.simulado
    carpeta_resultados = base_dir / args.out

    if carpeta_resultados.exists():
        shutil.rmtree(carpeta_resultados)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    if not archivo_observado.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo observado obligatorio: {archivo_observado.name}\n"
            "Para completar la Parte 2, copia validar_supermercado.csv en esta misma carpeta "
            "o usa --observado ruta/al/archivo.csv."
        )

    ejecutar_simulacion(base_dir, archivo_simulado, skip_run=args.skip_run)

    df_sim = leer_csv(archivo_simulado, "métricas simuladas")
    df_obs_base = leer_csv(archivo_observado, "datos observados base")

    print("\nColumnas simuladas disponibles:")
    print(df_sim.columns.tolist())
    print("\nColumnas observadas base disponibles:")
    print(df_obs_base.columns.tolist())

    df_obs_base_val, df_sim_base_val, metricas_base = preparar_bases_comparables(
        df_obs_base,
        df_sim,
        MAPA_COLUMNAS_BASE,
    )

    if not metricas_base:
        raise ValueError(
            "No se encontraron métricas base comunes. Revisa que validar_supermercado.csv tenga los nombres esperados."
        )

    print("\nMétricas base comunes:")
    for metrica in metricas_base:
        print(f"- {metrica}")

    crear_histogramas(
        df_obs_base_val,
        df_sim_base_val,
        metricas_base,
        carpeta_resultados / "histogramas_base",
    )

    tabla_base = validar_metricas(
        df_obs_base_val,
        df_sim_base_val,
        metricas_base,
        alpha=args.alpha,
    )
    tabla_base.round(6).to_csv(carpeta_resultados / "resultados_validacion_modelo_base.csv", index=False)
    imprimir_resumen_consola("VALIDACIÓN BASE", tabla_base)

    tabla_extra = None
    archivo_extra_usado: Optional[Path] = None

    if archivo_extra.exists():
        df_obs_extra = leer_csv(archivo_extra, "datos observados extra")
        print("\nColumnas observadas extra disponibles:")
        print(df_obs_extra.columns.tolist())

        df_obs_extra_val, df_sim_extra_val, metricas_extra = preparar_bases_comparables(
            df_obs_extra,
            df_sim,
            MAPA_COLUMNAS_EXTRA,
        )

        if metricas_extra:
            print("\nMétricas extra comunes:")
            for metrica in metricas_extra:
                print(f"- {metrica}")

            crear_histogramas(
                df_obs_extra_val,
                df_sim_extra_val,
                metricas_extra,
                carpeta_resultados / "histogramas_extra",
            )

            tabla_extra = validar_metricas(
                df_obs_extra_val,
                df_sim_extra_val,
                metricas_extra,
                alpha=args.alpha,
            )
            tabla_extra.round(6).to_csv(carpeta_resultados / "resultados_validacion_modelo_extra.csv", index=False)
            archivo_extra_usado = archivo_extra
            imprimir_resumen_consola("VALIDACIÓN EXTRA", tabla_extra)
        else:
            print("\nSe encontró metricas_extra.csv, pero no hubo columnas extra comunes con la simulación.")
    else:
        print("\nNo se encontró metricas_extra.csv. Se omite la validación extra.")

    escribir_resumen(
        carpeta_resultados / "resumen_validacion.txt",
        tabla_base,
        tabla_extra,
        archivo_observado,
        archivo_extra_usado,
    )

    if not args.no_zip:
        zip_path = base_dir / "resultados_parte_2_validacion.zip"
        comprimir_resultados(carpeta_resultados, zip_path)
        print(f"\nResultados comprimidos en: {zip_path}")

    print(f"\nResultados guardados en carpeta: {carpeta_resultados}")
    print("Archivos principales:")
    print(f"- {carpeta_resultados / 'resumen_validacion.txt'}")
    print(f"- {carpeta_resultados / 'resultados_validacion_modelo_base.csv'}")
    if tabla_extra is not None:
        print(f"- {carpeta_resultados / 'resultados_validacion_modelo_extra.csv'}")


if __name__ == "__main__":
    main()
