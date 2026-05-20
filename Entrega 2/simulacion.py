import simpy
import numpy as np
from pathlib import Path
from parametros import tasa_llegada_por_hora
from clases import Cliente, Almacen, Verduleria, Panaderia, Refrigerados


class Supermercado:
    def __init__(self, rng: np.random.Generator, hora_inicio: int = 9, hora_cierre: int = 21, activar_logs: bool = False, print_log_en_consola: bool = False, cantidad_de_dias_a_simular: int = 1):
        self.env = simpy.Environment()
        self.rng = rng
        self.hora_inicio = hora_inicio
        self.hora_cierre = hora_cierre
        self.activar_logs = activar_logs
        self.print_log_en_consola = print_log_en_consola
        self.cantidad_de_dias_a_simular = cantidad_de_dias_a_simular
        self.dia = 1
        self.log_path = Path("logs.txt")

        if self.activar_logs:
            self.log_file = self.log_path.open("w", encoding="utf-8")
        else:
            self.log_file = None

        self.clientes = []

        self.almacen = Almacen(self.env)
        self.verduleria = Verduleria(self.env)
        self.panaderia = Panaderia(self.env)
        self.refrigerados = Refrigerados(self.env)

    def registrar_evento(self, mensaje: str):
        dia = int(self.env.now // 24) + 1
        hora = int(self.env.now % 24)
        minutos = int((self.env.now % 1) * 60)
        segundos = int((self.env.now * 60) % 60)
        linea = f" [Dia {dia:02d} Hora {hora:02d}:{minutos:02d}:{segundos:02d}] {mensaje}"
        if self.activar_logs and self.log_file is not None:
            self.log_file.write(linea + "\n")
            self.log_file.flush()
        if self.print_log_en_consola:
            print(linea)

    def cerrar_logs(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def generar_personas_normales(self):
        while True:
            cliente_n = Cliente(self.env, "N")
            self.clientes.append(cliente_n)
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "N")
            tiempo_entre_llegadas = self.rng.exponential(1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)

    def generar_personas_preferenciales(self):
        while True:
            cliente_p = Cliente(self.env, "P")
            self.clientes.append(cliente_p)
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "P")
            tiempo_entre_llegadas = self.rng.exponential(1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)
