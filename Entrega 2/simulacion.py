import simpy
import numpy as np
from pathlib import Path
from parametros import tasa_llegada_por_hora
from clases import Cliente, Almacen, Reponedor, Sector, Verduleria, Panaderia, Refrigerados, Supermercado
from parametros import SECTOR, PROB_VISITAR_SECTOR, LIMITE_CAJA_AUTO, TIEMPO_DE_TRASLADO
from math import ceil


class Simulacion:
    def __init__(self, rng: np.random.Generator, activar_logs: bool = False,
                 print_log_en_consola: bool = False, hora_inicio: int = 9,
                 hora_cierre: int = 21, cantidad_de_dias_a_simular: int = 1,
                 nombre_log: str = "logs.txt"):
        self.env = simpy.Environment(initial_time=hora_inicio)
        self.rng = rng
        self.hora_inicio = hora_inicio
        self.hora_cierre = hora_cierre
        self.activar_logs = activar_logs
        self.print_log_en_consola = print_log_en_consola
        self.cantidad_de_dias_a_simular = cantidad_de_dias_a_simular
        self.dia = 1
        self.log_path = Path(nombre_log)

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
        segundos = int((self.env.now * 3600) % 60)
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
        while self.hora_inicio <= self.env.now < self.hora_cierre:
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "N")

            if tasa_llegada <= 0:
                break

            tiempo_entre_llegadas = self.rng.exponential(
                scale=1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)

            # Evita crear clientes si la llegada ocurrió después del cierre
            if self.env.now >= self.hora_cierre:
                break

            cliente_n = Cliente(self.env, "N")
            self.clientes.append(cliente_n)
            self.env.process(self.procesar_persona(cliente_n))

    def generar_personas_preferenciales(self):
        while self.hora_inicio <= self.env.now < self.hora_cierre:
            hora_actual = int(self.env.now) % 24
            tasa_llegada = tasa_llegada_por_hora(hora_actual, "P")

            if tasa_llegada <= 0:
                break

            tiempo_entre_llegadas = self.rng.exponential(
                scale=1 / tasa_llegada)
            yield self.env.timeout(tiempo_entre_llegadas)

            # Evita crear clientes si la llegada ocurrió después del cierre
            if self.env.now >= self.hora_cierre:
                break

            cliente_p = Cliente(self.env, "P")
            self.clientes.append(cliente_p)
            self.env.process(self.procesar_persona(cliente_p))

    def revisar_storage(self):
        while 9 <= self.env.now <= 21:
            yield self.env.timeout(0.5)
            if 9 < self.env.now < 21:
                for sector in [self.almacen, self.verduleria, self.panaderia, self.refrigerados]:
                    if sector.cuanto_stock() <= sector.stock.capacity * 0.7 and sector.solicitud_de_reponer is False:
                        sector.solicitud_de_reponer = True
                        self.registrar_evento(
                            f"Stock del sector {sector.nombre} bajo. Cantidad: {sector.cuanto_stock():.4f}. Reponiendo...")
                        self.env.process(self.solicitud_reponedor(sector))

    def calcular_cantidad_real_a_comprar(self, sector: Sector, cantidad_deseada: int) -> int:
        stock_disponible = int(sector.cuanto_stock())
        return max(0, min(int(cantidad_deseada), stock_disponible))

    def procesar_persona(self, cliente: Cliente):
        if not self.supermercado.entrar(cliente):
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) rechazado por capacidad al intentar entrar al supermercado.")
            return

        # formar lista
        sectores = []
        for s in SECTOR:
            if self.rng.random() < PROB_VISITAR_SECTOR[s]:
                sectores.append(s)
        cliente.sectores_a_visitar = sectores

        self.registrar_evento(
            f"Cliente {cliente.id_cliente} ({cliente.tipo}) entró al supermercado. Sectores a visitar en el siguiente orden: {cliente.sectores_a_visitar}")

        for sector in cliente.sectores_a_visitar:
            if sector == "almacen":
                tiempo_llegada_al_sector = self.env.now
                with self.almacen.request_espacio() as req:
                    yield req
                    cliente.tiempo_espera_sector["almacen"] = self.env.now - \
                        tiempo_llegada_al_sector
                    cliente.sector_actual = "almacen"
                    self.almacen.cantidad_clientes_visitados += 1
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Almacén.")

                    tiempo_espera = self.rng.lognormal(
                        mean=2.6, sigma=0.5) / 60

                    yield self.env.timeout(tiempo_espera)

                    cantidad_deseada = self.rng.negative_binomial(
                        n=4, p=0.25)

                    cantidad_items = self.calcular_cantidad_real_a_comprar(
                        self.almacen,
                        cantidad_deseada
                    )

                    if cantidad_items > 0:
                        yield self.almacen.sacar_items(cantidad_items)

                    self.almacen.cantidad_de_productos.append(
                        (self.env.now, float(self.almacen.cuanto_stock()))
                    )

                    cliente.cantidad_items += cantidad_items

                    utilidad = np.sum(self.rng.uniform(150,
                                                       650, size=cantidad_items))

                    cliente.utilidad += utilidad
                    self.almacen.utilidad_total += utilidad

                    if cantidad_items < cantidad_deseada:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) quiso comprar {cantidad_deseada} ítems en Almacén, pero solo pudo sacar {cantidad_items}. Disponibilidad actual: {self.almacen.cuanto_stock():.4f}."
                        )
                    else:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Almacén con {cantidad_items} ítems. Disponibilidad actual: {self.almacen.cuanto_stock():.4f}."
                        )

            elif sector == "verduleria":
                tiempo_llegada_al_sector = self.env.now
                with self.verduleria.request_espacio() as req:
                    yield req
                    cliente.tiempo_espera_sector["verduleria"] = (
                        self.env.now - tiempo_llegada_al_sector
                    )
                    cliente.sector_actual = "verduleria"

                    self.verduleria.cantidad_clientes_visitados += 1

                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Verdulería."
                    )

                    tiempo_espera = self.rng.gamma(shape=4, scale=3) / 60
                    yield self.env.timeout(tiempo_espera)

                    cantidad_deseada = self.rng.binomial(n=10, p=0.4)

                    cantidad_items = self.calcular_cantidad_real_a_comprar(
                        self.verduleria,
                        cantidad_deseada
                    )

                    if cantidad_items > 0:
                        yield self.verduleria.sacar_items(cantidad_items)

                    self.verduleria.cantidad_de_productos.append(
                        (self.env.now, float(self.verduleria.cuanto_stock()))
                    )

                    if cantidad_items > 0:
                        tiempo_llegada_a_balanza = self.env.now

                        with self.verduleria.request_balanza() as req_balanza:
                            yield req_balanza

                            cliente.tiempo_espera_balanza["verduleria"] = (
                                self.env.now - tiempo_llegada_a_balanza
                            )

                            tiempo_total_balanza = np.sum(self.rng.triangular(
                                10, 20, 30, size=cantidad_items)) / 3600
                            yield self.env.timeout(tiempo_total_balanza)

                            self.registrar_evento(
                                f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pesar en Verdulería."
                            )

                        utilidad = np.sum(self.rng.triangular(
                            50, 350, 1800, size=cantidad_items))
                        cliente.utilidad += utilidad
                        self.verduleria.utilidad_total += utilidad

                    cliente.cantidad_items += cantidad_items

                    if cantidad_items < cantidad_deseada:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) quiso comprar {cantidad_deseada} ítems en Verdulería, pero solo pudo sacar {cantidad_items}. Disponibilidad actual: {self.verduleria.cuanto_stock():.4f}."
                        )
                    else:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Verdulería con {cantidad_items} ítems. Disponibilidad actual: {self.verduleria.cuanto_stock():.4f}."
                        )

            elif sector == "panaderia":
                tiempo_llegada_al_sector = self.env.now
                with self.panaderia.request_espacio() as req:
                    yield req
                    cliente.tiempo_espera_sector["panaderia"] = self.env.now - \
                        tiempo_llegada_al_sector
                    cliente.sector_actual = "panaderia"

                    self.panaderia.cantidad_clientes_visitados += 1
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Panadería.")
                    tiempo_espera = self.rng.triangular(3, 5, 10) / 60

                    yield self.env.timeout(tiempo_espera)

                    cantidad_deseada = self.rng.poisson(lam=2) + 1

                    cantidad_items = self.calcular_cantidad_real_a_comprar(
                        self.panaderia,
                        cantidad_deseada
                    )

                    if cantidad_items > 0:
                        yield self.panaderia.sacar_items(cantidad_items)

                    self.panaderia.cantidad_de_productos.append(
                        (self.env.now, float(self.panaderia.cuanto_stock()))
                    )

                    if cantidad_items > 0:
                        tiempo_llegada_a_balanza = self.env.now

                        with self.panaderia.request_balanza() as req_balanza:
                            yield req_balanza

                            cliente.tiempo_espera_balanza["panaderia"] = (
                                self.env.now - tiempo_llegada_a_balanza
                            )

                            tiempo_total_balanza = np.sum(self.rng.triangular(
                                10, 20, 30, size=cantidad_items)) / 3600
                            yield self.env.timeout(tiempo_total_balanza)

                            self.registrar_evento(
                                f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pesar en Panadería."
                            )

                    cliente.cantidad_items += cantidad_items

                    utilidad = np.sum(self.rng.lognormal(
                        mean=6.0191, sigma=0.4245, size=cantidad_items))

                    cliente.utilidad += utilidad
                    self.panaderia.utilidad_total += utilidad

                    if cantidad_items < cantidad_deseada:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) quiso comprar {cantidad_deseada} ítems en Panadería, pero solo pudo sacar {cantidad_items}. Disponibilidad actual: {self.panaderia.cuanto_stock():.4f}."
                        )
                    else:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Panadería con {cantidad_items} ítems. Disponibilidad actual: {self.panaderia.cuanto_stock():.4f}."
                        )

            elif sector == "refrigerados":
                tiempo_llegada_al_sector = self.env.now
                with self.refrigerados.request_espacio() as req:
                    yield req
                    cliente.tiempo_espera_sector["refrigerados"] = self.env.now - \
                        tiempo_llegada_al_sector
                    cliente.sector_actual = "refrigerados"
                    self.refrigerados.cantidad_clientes_visitados += 1
                    self.registrar_evento(
                        f"Cliente {cliente.id_cliente} ({cliente.tipo}) ingresó al sector Refrigerados.")
                    tiempo_espera = self.rng.weibull(a=2) * 10 / 60
                    yield self.env.timeout(tiempo_espera)
                    cantidad_deseada = self.rng.poisson(lam=5) + 1
                    cantidad_items = self.calcular_cantidad_real_a_comprar(
                        self.refrigerados,
                        cantidad_deseada
                    )

                    if cantidad_items > 0:
                        yield self.refrigerados.sacar_items(cantidad_items)

                    self.refrigerados.cantidad_de_productos.append(
                        (self.env.now, float(self.refrigerados.cuanto_stock()))
                    )

                    if cantidad_items < cantidad_deseada:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) quiso comprar {cantidad_deseada} ítems en Refrigerados, pero solo pudo sacar {cantidad_items}. Disponibilidad actual: {self.refrigerados.cuanto_stock():.4f}."
                        )
                    else:
                        self.registrar_evento(
                            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del sector Refrigerados con {cantidad_items} ítems. Disponibilidad actual: {self.refrigerados.cuanto_stock():.4f}."
                        )

                    cliente.cantidad_items += cantidad_items

                    utilidad = np.sum(self.rng.gamma(shape=5,
                                                     scale=180, size=cantidad_items))

                    cliente.utilidad += utilidad
                    self.refrigerados.utilidad_total += utilidad

        if len(cliente.sectores_a_visitar) == 0:
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) no visitó ningún sector, tomó 1 ítem de Almacén y se dirigió directamente a la caja.")

            item = self.calcular_cantidad_real_a_comprar(self.almacen, 1)
            if item > 0:
                yield self.almacen.sacar_items(1)
            self.almacen.cantidad_de_productos.append(
                (self.env.now, float(self.almacen.cuanto_stock()))
            )

            cliente.cantidad_items += 1
            utilidad = self.rng.uniform(150, 650)
            cliente.utilidad += utilidad
            self.almacen.utilidad_total += utilidad

        # Elección de caja según reglas: autoservicio solo si cantidad_items <= LIMITE_CAJA_AUTO
        caja, idx, cantidad = self.supermercado.elegir_caja(
            cliente, self.rng, self.registrar_evento)
        llegada_caja = self.env.now
        if caja == "auto":
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja automática.")
            with self.supermercado.request_caja_auto() as req:
                yield req
                self.supermercado.caja_clientes_auto += 1
                cliente.tiempo_espera_caja = self.env.now - llegada_caja
                self.supermercado.tiempo_espera_caja_auto.append(
                    cliente.tiempo_espera_caja)

                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a pagar en caja automática.")
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                    tiempo_items = self.rng.lognormal(
                        mean=2.2, sigma=0.5) * cliente.cantidad_items / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=60) / 3600
                    tiempo_items = self.rng.gamma(
                        shape=4, scale=2) * cliente.cantidad_items / 3600
                yield self.env.timeout(tiempo_items)
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja automática.")
        elif caja == "preferencial":
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja preferencial {idx}.")
            with self.supermercado.request_caja_preferencial(idx, cliente) as req:
                yield req
                self.supermercado.caja_clientes_preferencial += 1
                cliente.tiempo_espera_caja = self.env.now - llegada_caja
                self.supermercado.tiempo_espera_caja_preferencial.append(
                    cliente.tiempo_espera_caja)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a escanear en caja preferencial {idx}.")
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                    tiempo_items = self.rng.lognormal(
                        mean=2.2, sigma=0.5) * cliente.cantidad_items / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=60) / 3600
                    tiempo_items = self.rng.gamma(
                        shape=4, scale=2) * cliente.cantidad_items / 3600
                yield self.env.timeout(tiempo_items)
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja preferencial {idx}.")
        else:
            self.registrar_evento(
                f"Cliente {cliente.id_cliente} ({cliente.tipo}) eligió caja normal {idx}.")
            with self.supermercado.request_caja_normal(idx, cliente) as req:
                yield req
                self.supermercado.caja_clientes_normal += 1
                cliente.tiempo_espera_caja = self.env.now - llegada_caja
                self.supermercado.tiempo_espera_caja_normal.append(
                    cliente.tiempo_espera_caja)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) comenzó a escanear en caja normal {idx}.")
                if cliente.tipo == "P":
                    tiempo_pago = self.rng.weibull(a=1.2) * 190 / 3600
                    tiempo_items = self.rng.lognormal(
                        mean=2.2, sigma=0.5) * cliente.cantidad_items / 3600
                else:
                    tiempo_pago = self.rng.exponential(scale=60) / 3600
                    tiempo_items = self.rng.gamma(
                        shape=4, scale=2) * cliente.cantidad_items / 3600
                yield self.env.timeout(tiempo_items)
                yield self.env.timeout(tiempo_pago)
                self.registrar_evento(
                    f"Cliente {cliente.id_cliente} ({cliente.tipo}) terminó de pagar en caja normal {idx}.")

        self.registrar_evento(
            f"Cliente {cliente.id_cliente} ({cliente.tipo}) salió del supermercado. Con una utilidad de ${cliente.utilidad:.2f}.")

        cliente.hora_salida = self.env.now
        cliente.tiempo_de_permanencia = cliente.hora_salida - cliente.hora_llegada
        self.supermercado.salir(cliente)

    def solicitud_reponedor(self, sector):
        reponedor = yield self.supermercado.request_reponedor()

        self.registrar_evento(
            f"Reponedor {reponedor.id_reponedor} asignado para reponer el sector {sector.nombre}."
        )

        try:
            yield self.env.process(self.procesar_reponedor(reponedor, sector))
        finally:
            yield self.supermercado.devolver_reponedor(reponedor)

    def procesar_reponedor(self, reponedor: Reponedor, sector: Sector):

        reponedor.estado = "llendo"
        reponedor.sector_actual = sector.nombre

        yield self.env.timeout(TIEMPO_DE_TRASLADO / 60)  # minutos a horas
        reponedor.estado = "esperando reponer"

        solicitudes = [sector.request_espacio() for _ in range(4)]
        yield simpy.events.AllOf(self.env, solicitudes)
        try:
            reponedor.estado = "reponiendo"

            proporcion_reponer = 0.25 * self.rng.beta(a=2, b=5)
            cantidad_teorica = ceil(
                sector.stock.capacity * proporcion_reponer)
            espacio_disponible = sector.stock.capacity - sector.cuanto_stock()
            cantidad_real_a_reponer = int(
                min(cantidad_teorica, espacio_disponible))

            if sector.nombre == "almacen":
                tiempo_reponer = self.rng.lognormal(mean=2.9, sigma=0.3) / 60
            elif sector.nombre == "verduleria":
                tiempo_reponer = self.rng.gamma(shape=3, scale=5) / 60
            elif sector.nombre == "refrigerados":
                tiempo_reponer = self.rng.weibull(a=1.5) * 13.3 / 60
            elif sector.nombre == "panaderia":
                tiempo_reponer = self.rng.exponential(scale=8) / 60

            yield self.env.timeout(tiempo_reponer)

            yield sector.reponer_items(cantidad_real_a_reponer)

            sector.cantidad_de_productos.append(
                (self.env.now, float(sector.cuanto_stock()))
            )
        finally:
            reponedor.estado = "descansando"
            reponedor.sector_actual = None
            for req in solicitudes:
                sector.release_espacio(req)
            sector.solicitud_de_reponer = False
            self.registrar_evento(
                f"Reponedor {reponedor.id_reponedor} terminó de reponer el sector {sector.nombre}. Cantidad repuesta: {cantidad_real_a_reponer}. Stock actual: {sector.cuanto_stock():.4f}.")

    def calcular_producto_promedio(self, sector: Sector):
        historial = sorted(sector.cantidad_de_productos, key=lambda x: x[0])

        inicio = self.hora_inicio
        cierre = self.hora_cierre

        area_total = 0.0
        tiempo_anterior = inicio

        # Al inicio del día se asume stock lleno
        nivel_anterior = float(sector.stock.capacity)

        for tiempo, nivel in historial:
            if tiempo < inicio:
                continue

            if tiempo > cierre:
                break

            tiempo_transcurrido = tiempo - tiempo_anterior
            area_total += nivel_anterior * tiempo_transcurrido

            tiempo_anterior = tiempo
            nivel_anterior = nivel

        if tiempo_anterior < cierre:
            area_total += nivel_anterior * (cierre - tiempo_anterior)

        return area_total / (cierre - inicio)

    def recolectar_estadisticas(self):
        # Aquí se pueden recolectar estadísticas al finalizar la simulación

        utilidades = 0
        cantidad_clientes_atendidos = 0
        tiempo_espera_almacen = []
        tiempo_espera_verduleria = []
        tiempo_espera_panaderia = []
        tiempo_espera_refrigerados = []
        tiempo_espera_balanza_verduleria = []
        tiempo_espera_balanza_panaderia = []
        tiempo_permanencia_normales = []
        tiempo_permanencia_preferenciales = []

        for cliente in self.clientes:
            if cliente.estado == "terminado":
                utilidades += cliente.utilidad
                cantidad_clientes_atendidos += 1
            if "almacen" in cliente.sectores_a_visitar and cliente.tiempo_espera_sector["almacen"] is not None:
                tiempo_espera_almacen.append(
                    cliente.tiempo_espera_sector["almacen"])
            if "verduleria" in cliente.sectores_a_visitar and cliente.tiempo_espera_sector["verduleria"] is not None:
                tiempo_espera_verduleria.append(
                    cliente.tiempo_espera_sector["verduleria"])
            if "panaderia" in cliente.sectores_a_visitar and cliente.tiempo_espera_sector["panaderia"] is not None:
                tiempo_espera_panaderia.append(
                    cliente.tiempo_espera_sector["panaderia"])
            if "refrigerados" in cliente.sectores_a_visitar and cliente.tiempo_espera_sector["refrigerados"] is not None:
                tiempo_espera_refrigerados.append(
                    cliente.tiempo_espera_sector["refrigerados"])
            if "verduleria" in cliente.sectores_a_visitar and cliente.tiempo_espera_balanza["verduleria"] is not None:
                tiempo_espera_balanza_verduleria.append(
                    cliente.tiempo_espera_balanza["verduleria"])
            if "panaderia" in cliente.sectores_a_visitar and cliente.tiempo_espera_balanza["panaderia"] is not None:
                tiempo_espera_balanza_panaderia.append(
                    cliente.tiempo_espera_balanza["panaderia"])
            if cliente.tiempo_de_permanencia is not None:
                if cliente.tipo == "N":
                    tiempo_permanencia_normales.append(
                        cliente.tiempo_de_permanencia)
                elif cliente.tipo == "P":
                    tiempo_permanencia_preferenciales.append(
                        cliente.tiempo_de_permanencia)

        tiempo_espera_caja_normal_avg = np.mean(
            self.supermercado.tiempo_espera_caja_normal)
        tiempo_espera_caja_preferencial_avg = np.mean(
            self.supermercado.tiempo_espera_caja_preferencial)
        tiempo_espera_caja_auto_avg = np.mean(
            self.supermercado.tiempo_espera_caja_auto)

        avg_productos_almacen = self.calcular_producto_promedio(self.almacen)
        avg_productos_verduleria = self.calcular_producto_promedio(
            self.verduleria)
        avg_productos_panaderia = self.calcular_producto_promedio(
            self.panaderia)
        avg_productos_refrigerados = self.calcular_producto_promedio(
            self.refrigerados)

        gastos = 14 * 27000  # Costo de 14 empleados a $27.000 cada uno
        # Costo por minuto extra de apertura
        tiempo_extra = int(
            (max(0, self.env.now - self.hora_cierre)) * 60) * 150

        return {
            "utilidades totales": utilidades - gastos - tiempo_extra,
            "Promedio productos verdureria": avg_productos_verduleria,
            "Promedio productos panaderia": avg_productos_panaderia,
            "Promedio productos almacen": avg_productos_almacen,
            "Promedio productos refrigerados": avg_productos_refrigerados,
            "Promedio tiempo espera almacen": np.mean(tiempo_espera_almacen) * 60,
            "Promedio tiempo espera verduleria": np.mean(tiempo_espera_verduleria) * 60,
            "Promedio tiempo espera panaderia": np.mean(tiempo_espera_panaderia) * 60,
            "Promedio tiempo espera refrigerados": np.mean(tiempo_espera_refrigerados) * 60,
            "Promedio tiempo espera balanza verduleria": np.mean(tiempo_espera_balanza_verduleria) * 60,
            "Promedio tiempo espera balanza panaderia": np.mean(tiempo_espera_balanza_panaderia) * 60,
            "Promedio tiempo espera caja normal": tiempo_espera_caja_normal_avg * 60,
            "Promedio tiempo espera caja preferencial": tiempo_espera_caja_preferencial_avg * 60,
            "Promedio tiempo espera caja automática": tiempo_espera_caja_auto_avg * 60,
            "Promedio tiempo de permanencia clientes normales": np.mean(tiempo_permanencia_normales) * 60,
            "Promedio tiempo de permanencia clientes preferenciales": np.mean(tiempo_permanencia_preferenciales) * 60,
            "Cantidad total de clientes atendidos": cantidad_clientes_atendidos,
            "utilidad verduleria": self.verduleria.utilidad_total,
            "utilidad panaderia": self.panaderia.utilidad_total,
            "utilidad almacen": self.almacen.utilidad_total,
            "utilidad refrigerados": self.refrigerados.utilidad_total,
            "cantidad clientes visita verduleria": self.verduleria.cantidad_clientes_visitados,
            "cantidad clientes visita panaderia": self.panaderia.cantidad_clientes_visitados,
            "cantidad clientes visita almacen": self.almacen.cantidad_clientes_visitados,
            "cantidad clientes visita refrigerados": self.refrigerados.cantidad_clientes_visitados,
            "cantidad clientes caja automática": self.supermercado.caja_clientes_auto,
            "cantidad clientes caja preferencial": self.supermercado.caja_clientes_preferencial,
            "cantidad clientes caja normal": self.supermercado.caja_clientes_normal,
            "cantidad clientes rechazados por capacidad": self.supermercado.clientes_rechazados_por_capacidad
        }

    def crear_log_estadisticas(self, estadisticas: dict, nombre_archivo: str = "estadisticas_simulacion.txt"):
        """Guarda en un archivo de texto el resumen devuelto por recolectar_estadisticas."""
        lineas = [
            "",
            "Resumen de estadisticas de la simulacion",
            f"Dia simulado: {self.dia}",
            f"Tiempo total simulado: {self.env.now:.4f}",
            "-" * 50,
        ]

        for clave, valor in estadisticas.items():
            if isinstance(valor, (np.integer, int)):
                valor_formateado = f"{int(valor)}"
            elif isinstance(valor, (np.floating, float)):
                valor_formateado = f"{float(valor):.2f}"
            else:
                valor_formateado = str(valor)
            lineas.append(f"{clave}: {valor_formateado}")

        contenido = "\n".join(lineas) + "\n"

        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(contenido)
            self.log_file.flush()
            return self.log_path

        ruta = self.log_path if nombre_archivo == "estadisticas_simulacion.txt" else Path(
            nombre_archivo)
        with ruta.open("a", encoding="utf-8") as archivo:
            archivo.write(contenido)

        return ruta

    def generar_reporte_estadisticas(self, nombre_archivo: str = "estadisticas_simulacion.txt"):
        """Calcula las estadisticas y las guarda en un archivo."""
        estadisticas = self.recolectar_estadisticas()
        return self.crear_log_estadisticas(estadisticas, nombre_archivo=nombre_archivo)

    def ejecutar(self):
        self.env.process(self.generar_personas_normales())
        self.env.process(self.generar_personas_preferenciales())
        self.env.process(self.revisar_storage())

        self.env.run()

        estadisticas = self.recolectar_estadisticas()

        if self.activar_logs:
            self.crear_log_estadisticas(estadisticas)

        self.cerrar_logs()

        return estadisticas
