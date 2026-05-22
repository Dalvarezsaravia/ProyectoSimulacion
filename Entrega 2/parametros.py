SECTOR = ["verduleria", "panaderia", "almacen", "refrigerados"]

# Capacidad total del sistema
C_MAX = 200

# Capacidad de clientes por sector
CAPACIDAD_SECTOR = {
    "almacen": 50,
    "verduleria": 30,
    "panaderia": 15,
    "refrigerados": 20,
}

# Capacidad máxima de ítems por sector (K_s)
CAPACIDAD_SECTOR_ITEM = {
    "almacen": 12000,
    "verduleria": 1800,
    "panaderia": 1200,
    "refrigerados": 3500,
}

# Número de balanzas por sector
BALANZAS_POR_SECTOR = {"verduleria": 3, "panaderia": 2}

# Probabilidad de visitar cada sector (P_s)
PROB_VISITAR_SECTOR = {
    "almacen": 0.95,
    "verduleria": 0.75,
    "panaderia": 0.60,
    "refrigerados": 0.70,
}

# Tiempo de traslado a bodega para reposición (F, minutos)
TIEMPO_DE_TRASLADO = 10

# Cantidad de cajas por tipo: autoservicio (a), preferencial (p), normal (n)
CAJAS_POR_TIPO = {"auto": 4, "preferencial": 1, "normal": 3}

# Límite de ítems para usar autoservicio
LIMITE_CAJA_AUTO = 30

CLIENTE_ORDEN = {"P": 1, "N": 2}

# Tasas de llegada por hora (esperadas por hora) por tipo de cliente
# Clave: hora de inicio del bloque (entero, 24h). Valores: 'P' = preferente, 'N' = normal
TASAS_LLEGADA_POR_HORA = {
    9:  {"P": 12, "N": 28},
    10: {"P": 18, "N": 42},
    11: {"P": 17, "N": 58},
    12: {"P": 12, "N": 78},
    13: {"P": 12, "N": 98},
    14: {"P": 8,  "N": 82},
    15: {"P": 9,  "N": 56},
    16: {"P": 12, "N": 68},
    17: {"P": 13, "N": 92},
    18: {"P": 11, "N": 134},
    19: {"P": 10, "N": 145},
    20: {"P": 4,  "N": 96},
}


def tasa_llegada_por_hora(hora: int, tipo: str) -> float:
    """Devuelve la tasa de llegada (esperada por hora) para una `hora` y tipo `tipo` ('P' o 'N').

    Si la hora no está en la tabla devuelve 0.
    """
    tipos = TASAS_LLEGADA_POR_HORA.get(hora)
    if tipos is None:
        return 0.0
    return float(tipos.get(tipo.upper(), 0.0))
