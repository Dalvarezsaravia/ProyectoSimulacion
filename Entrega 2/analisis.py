import pandas as pd
import numpy as np
import scipy.stats as st


# PARA EJECUTAR ESTE CÓDIGO SE TIENE QUE HABER EJECUTADO EL main.py DE LA PARTE 1, sino no funciona. 
# y tiene que estar en la misma carpeta que los archivos generados en la parte 1.

# cargar los datos de las metricas obtenidas en la parte 1
def cargar_datos(archivo_csv="metricas_replicas.csv"):
    try:
        df = pd.read_csv(archivo_csv)
        print(f"Datos cargados exitosamente: {len(df)} réplicas encontradas.\n")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_csv}.")
        return None
    
# función para calcular Intervalo de Confianza para los promedios
def calcular_ic_media(datos, alpha=0.05):
    n = len(datos)
    x_barra = np.mean(datos)
    s_cuadrado = np.var(datos, ddof=1)
    t = st.t.ppf(1 - alpha / 2, df=n - 1) #valor t
    
    largo_medio = t * np.sqrt(s_cuadrado / n) 
    lim_inf = x_barra - largo_medio
    lim_sup = x_barra + largo_medio

    return x_barra, largo_medio, lim_inf, lim_sup

# funcion para calcular el ic para los percentiles
def calcular_ic_percentil(datos, q, alpha=0.05):
    n = len(datos)
    datos_ordenados = np.sort(datos)

    idx_est = int(np.floor(n * q))  
    estimador = datos_ordenados[idx_est]
    
    z = st.norm.ppf(1 - alpha / 2)
    raiz = np.sqrt(n * q * (1 - q))

    # indices r y s para los límites del intervalo
    r = int(np.ceil(n * q - z * raiz))
    s = int(np.ceil(n * q + z * raiz))
    
    # para que los índices no se salgan de la lista (0 a n-1)
    r_idx = max(0, r - 1)
    s_idx = min(n - 1, s - 1)
    
    lim_inf = datos_ordenados[r_idx]
    lim_sup = datos_ordenados[s_idx]
    
    return estimador, lim_inf, lim_sup


# aplicaión a métricas
def aplicacion_metricas(df): 

    print(" ANÁLISIS DE OUTPUT") #usé IA para que me ayudara a hacer los print más bonitos

    # i) Promedio de productos por sector
    print("\n i) Promedio de productos por sector (Media)")
    sectores = ["verdureria", "panaderia", "almacen", "refrigerados"]
    for s in sectores:
        columna = f"Promedio productos {s}" 
        media, largo_medio, inf, sup = calcular_ic_media(df[columna])
        print(f"{s.capitalize():<12}: {media:>8.2f} +/- {largo_medio:>5.2f}  IC: [{inf:>7.2f}, {sup:>7.2f}]")

    # ii) Tiempos de espera
    print("\n ii) Tiempos de espera promedio (minutos)")
    columnas_espera = [
        "Promedio tiempo espera almacen",
        "Promedio tiempo espera verduleria",
        "Promedio tiempo espera panaderia",
        "Promedio tiempo espera refrigerados",
        "Promedio tiempo espera balanza verduleria",
        "Promedio tiempo espera balanza panaderia",
        "Promedio tiempo espera caja normal",
        "Promedio tiempo espera caja preferencial",
        "Promedio tiempo espera caja automática"
    ]
    for col in columnas_espera:
        media, largo_medio, inf, sup = calcular_ic_media(df[col])
        nombre_limpio = col.replace("Promedio tiempo espera ", "").capitalize()
        print(f"{nombre_limpio:<18}: {media:>,.4f} +/- {largo_medio:>,.4f}  IC: [{inf:>,.2f}, {sup:>,.2f}]")

    # iii) Tiempos de permanencia por tipo
    print("\niii) Promedio de tiempo de permanencia por tipo (minutos)")
    columnas_permanencia = [
        "Promedio tiempo de permanencia clientes normales",
        "Promedio tiempo de permanencia clientes preferenciales"
    ]   
    for col in columnas_permanencia:
        media, largo_medio, inf, sup = calcular_ic_media(df[col])
        nombre_limpio = col.replace("Promedio tiempo de permanencia clientes ", "").capitalize()
        print(f"{nombre_limpio:<15}: {media:>,.2f} +/- {largo_medio:>,.2f}  IC: [{inf:>,.2f}, {sup:>,.2f}]")

    # iv) Utilidades (Percentiles)
    print("\n iv) Utilidad del Supermercado (Percentiles)")
    utilidades = df["utilidades totales"].values
    
    p25, inf25, sup25 = calcular_ic_percentil(utilidades, 0.25)
    p50, inf50, sup50 = calcular_ic_percentil(utilidades, 0.50)
    p75, inf75, sup75 = calcular_ic_percentil(utilidades, 0.75)
    
    print(f"Percentil 25 : ${p25:,.0f}  IC: [${inf25:,.0f}, ${sup25:,.0f}]")
    print(f"Mediana (p50): ${p50:,.0f}  IC: [${inf50:,.0f}, ${sup50:,.0f}]")
    print(f"Percentil 75 : ${p75:,.0f}  IC: [${inf75:,.0f}, ${sup75:,.0f}]")

if __name__ == "__main__":
    df_resultados = cargar_datos()
    if df_resultados is not None:
        aplicacion_metricas(df_resultados)













