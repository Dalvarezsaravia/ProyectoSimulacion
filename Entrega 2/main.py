from simulacion import Simulacion
from clases import Cliente, Reponedor
import numpy as np
import csv


N_REPLICAS = 200
SEED_BASE = 2401050808


def correr_replica(numero_replica: int, seed: int, activar_logs: bool = False):
    # Reiniciar contadores para que cada réplica parta desde Cliente 1 y Reponedor 1
    Cliente.id = 1
    Reponedor.id = 1

    rng = np.random.default_rng(seed=seed)

    sim = Simulacion(
        rng,
        activar_logs=activar_logs,
        print_log_en_consola=False,
        nombre_log="logs.txt"
    )

    estadisticas = sim.ejecutar()

    fila = {
        "replica": numero_replica,
        "seed": seed,
    }
    fila.update(estadisticas)

    return fila


def guardar_resultados_csv(resultados, nombre_archivo="metricas_replicas.csv"):
    if len(resultados) == 0:
        return

    columnas = list(resultados[0].keys())

    with open(nombre_archivo, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(resultados)


def main():
    resultados = []
    print(f"Iniciando simulación con {N_REPLICAS} réplicas...")

    for i in range(N_REPLICAS):
        print(f"Corriendo réplica {i + 1} de {N_REPLICAS}")
        numero_replica = i + 1
        seed = SEED_BASE + i

        # Solo la primera réplica genera logs.txt.
        # Las demás solo guardan estadísticas en el CSV.
        activar_logs = (i == 0)

        resultado = correr_replica(
            numero_replica=numero_replica,
            seed=seed,
            activar_logs=activar_logs
        )

        resultados.append(resultado)

    guardar_resultados_csv(resultados)

    print(f"Simulación terminada. Se generaron {N_REPLICAS} réplicas.")
    print("Log de muestra guardado en: logs.txt")
    print("Métricas guardadas en: metricas_replicas.csv")


if __name__ == "__main__":
    main()
