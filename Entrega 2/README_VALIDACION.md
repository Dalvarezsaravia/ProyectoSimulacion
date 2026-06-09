# Validación Parte 2

Este paquete agrega la validación de la Parte 2 al repositorio del proyecto.

## Archivos que se deben subir al repositorio

- `validacion_parte2.py`
- `requirements_validacion.txt` (opcional, pero recomendado)

## Archivo obligatorio que debe estar disponible para correr la validación

- `validar_supermercado.csv`

Este archivo corresponde a los 200 días observados entregados por la profesora. Debe estar en la misma carpeta que `main.py`, `simulacion.py`, `clases.py` y `parametros.py`, o se debe pasar su ruta con `--observado`.

## Archivo opcional

- `metricas_extra.csv`

Si este archivo está presente, el script ejecuta también la validación extra. Si no está, se omite automáticamente.

## Instalación de dependencias

Desde la carpeta del proyecto:

```bash
pip install -r requirements_validacion.txt
```

## Ejecución recomendada

Desde la carpeta donde están `main.py`, `simulacion.py`, `clases.py` y `parametros.py`:

```bash
python validacion_parte2.py
```

El script ejecuta `main.py`, genera `metricas_replicas.csv`, compara contra `validar_supermercado.csv`, genera histogramas, aplica Mann-Whitney, calcula intervalos de confianza y deja los resultados en:

```text
resultados_parte_2_validacion/
```

También genera:

```text
resultados_parte_2_validacion.zip
```

## Si ya existe `metricas_replicas.csv`

Para no volver a correr las 200 réplicas:

```bash
python validacion_parte2.py --skip-run
```

## Rutas personalizadas

```bash
python validacion_parte2.py --observado ruta/validar_supermercado.csv --extra ruta/metricas_extra.csv
```

## Criterio estadístico usado

Se usa alpha = 0.05. Una métrica se considera validada si cumple simultáneamente:

1. Test U de Mann-Whitney: `p-value >= 0.05`.
2. Intervalo de confianza al 95% de la diferencia de medias simulada menos observada contiene 0.
