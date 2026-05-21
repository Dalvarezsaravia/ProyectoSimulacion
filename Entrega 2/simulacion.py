import simpy
import numpy as np
from pathlib import Path
from parametros import tasa_llegada_por_hora
from clases import Cliente, Almacen, Verduleria, Panaderia, Refrigerados, Supermercado
from parametros import SECTOR, PROB_VISITAR_SECTOR, LIMITE_CAJA_AUTO


class Simulacion:
    def __init__(self, rng: np.random.Generator, activar_logs: bool = False,
                 print_log_en_consola: bool = False, hora_inicio: int = 9, hora_cierre: int = 21,  cantidad_de_dias_a_simular: int = 1):
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
        self.supermercado = Supermercado(self.env)

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
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "N")
            tiempo_entre_llegadas = self.rng.exponential(1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)
            cliente_n = Cliente(self.env, "N")
            self.clientes.append(cliente_n)
            self.env.process(self.procesar_persona(cliente_n))

    def generar_personas_preferenciales(self):
        while True:
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "P")
            tiempo_entre_llegadas = self.rng.exponential(1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)
            cliente_p = Cliente(self.env, "P")
            self.clientes.append(cliente_p)
            self.env.process(self.procesar_persona(cliente_p))

    def procesar_persona(self, cliente: Cliente):
        if not self.supermercado.entrar(cliente):
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) rechazado por capacidad al intentar entrar al supermercado.")
            return

        # formar lista
        sectores = [s for s in SECTOR if self.rng.random() <
                    PROB_VISITAR_SECTOR[s]]
        # mezclar reproducible con Generator
        self.rng.shuffle(sectores)
        # Aqui se mezclan los sectores a visitar para cada cliente, para que no siempre visiten en el mismo orden
        cliente.sectores_a_visitar = sectores
        self.registrar_evento(
            f"Cliente {cliente.id_cliente} ({cliente.tipo}) entró al supermercado. Sectores a visitar en el siguiente orden: {cliente.sectores_a_visitar}")

        for sector in cliente.sectores_a_visitar:
            if sector == "almacen":
                with self.almacen.request_espacio() as req:
                    yield req
                    cliente.sector_actual = "almacen"
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Almacén.")
                    tiempo_espera = self.rng.lognormal(
                        mean=2.6, sigma=0.5) / 60
                    yield self.env.timeout(tiempo_espera)
                    cantidad_items = self.rng.negative_binomial(n=4, p=0.25)
                    yield self.almacen.sacar_items(cantidad_items)
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Almacén con {cantidad_items} ítems.")
                    cliente.cantidad_items += cantidad_items

            elif sector == "verduleria":
                with self.verduleria.request_espacio() as req:
                    yield req
                    cliente.sector_actual = "verduleria"
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Verdulería.")
                    tiempo_espera = self.rng.gamma(shape=4, scale=3) / 60
                    yield self.env.timeout(tiempo_espera)
                    cantidad_items = self.rng.binomial(n=10, p=0.4)
                    yield self.verduleria.sacar_items(cantidad_items)
                    with self.verduleria.request_balanza() as req_balanza:
                        yield req_balanza
                        for _ in range(cantidad_items):
                            tiempo_en_balanza = self.rng.triangular(
                                left=10, mode=20, right=30) / 3600
                            yield self.env.timeout(tiempo_en_balanza)
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pesar en Verdulería.")
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Verdulería con {cantidad_items} ítems.")
                    cliente.cantidad_items += cantidad_items

            elif sector == "panaderia":
                with self.panaderia.request_espacio() as req:
                    yield req
                    cliente.sector_actual = "panaderia"
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Panadería.")
                    tiempo_espera = self.rng.triangular(
                        left=3, mode=5, right=10) / 60
                    yield self.env.timeout(tiempo_espera)
                    cantidad_items = self.rng.poisson(lam=2) + 1
                    yield self.panaderia.sacar_items(cantidad_items)

                    with self.panaderia.request_balanza() as req_balanza:
                        yield req_balanza
                        for _ in range(cantidad_items):
                            tiempo_en_balanza = self.rng.triangular(
                                left=10, mode=20, right=30) / 3600
                            yield self.env.timeout(tiempo_en_balanza)
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pesar en Panadería.")
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Panadería con {cantidad_items} ítems.")
                    cliente.cantidad_items += cantidad_items

            elif sector == "refrigerados":
                with self.refrigerados.request_espacio() as req:
                    yield req
                    cliente.sector_actual = "refrigerados"
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Refrigerados.")
                    tiempo_espera = self.rng.weibull(a=2) * 10 / 60
                    yield self.env.timeout(tiempo_espera)
                    cantidad_items = self.rng.poisson(lam=5) + 1
                    yield self.refrigerados.sacar_items(cantidad_items)
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Refrigerados con {cantidad_items} ítems.")
                    cliente.cantidad_items += cantidad_items

        # Elección de caja según reglas: autoservicio solo si cantidad_items <= LIMITE_CAJA_AUTO
        caja, idx, cantidad = self.supermercado.elegir_caja(cliente, self.rng)
        if caja == "auto":
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja automática.")
            with self.supermercado.request_caja_auto() as req:
                yield req
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a pagar en caja automática.")
                for _ in range(cliente.cantidad_items):
                    if cliente.tipo == "P":
                        tiempo_item = self.rng.lognormal(
                            mean=2.2, sigma=0.5) / 3600
                    else:
                        tiempo_item = self.rng.gamma(shape=4, scale=2) / 3600
                    yield self.env.timeout(tiempo_item)
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=1/60) / 3600
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja automática.")
        elif caja == "preferencial":
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja preferencial {idx}.")
            with self.supermercado.request_caja_preferencial(idx, cliente) as req:
                yield req
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a escanear en caja preferencial {idx}.")
                for _ in range(cliente.cantidad_items):
                    if cliente.tipo == "P":
                        tiempo_item = self.rng.lognormal(
                            mean=2.2, sigma=0.5) / 3600
                    else:
                        tiempo_item = self.rng.gamma(shape=4, scale=2) / 3600
                    yield self.env.timeout(tiempo_item)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de escanear en caja preferencial {idx}.")
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=1/60) / 3600
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja preferencial {idx}.")
        else:
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja normal {idx}.")
            with self.supermercado.request_caja_normal(idx, cliente) as req:
                yield req
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a escanear en caja normal {idx}.")
                for _ in range(cliente.cantidad_items):
                    if cliente.tipo == "P":
                        tiempo_item = self.rng.lognormal(
                            mean=2.2, sigma=0.5) / 3600
                    else:
                        tiempo_item = self.rng.gamma(shape=4, scale=2) / 3600
                    yield self.env.timeout(tiempo_item)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de escanear en caja normal {idx}.")
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=1/60) / 3600
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja normal {idx}.")

        self.registrar_evento(
            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del supermercado.")

        cliente.hora_salida = self.env.now
        self.supermercado.salir(cliente)
