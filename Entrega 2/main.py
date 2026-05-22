from simulacion import Simulacion
import numpy as np


def main():
    rng = np.random.default_rng(seed=240105)
    sim = Simulacion(rng, activar_logs=True, print_log_en_consola=False)
    sim.ejecutar()


if __name__ == "__main__":
    main()
