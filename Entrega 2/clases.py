import simpy
import numpy as np
from parametros import (
    CAPACIDAD_SECTOR,
    CAPACIDAD_SECTOR_ITEM,
    BALANZAS_POR_SECTOR,
    CAJAS_POR_TIPO,
    C_MAX,
    CLIENTE_ORDEN,
    LIMITE_CAJA_AUTO,
)


class Cliente:
    id = 1

    def __init__(self, env: simpy.Environment, tipo: str):
        self.env = env
        self.tipo = tipo
        self.id_cliente = Cliente.id
        Cliente.id += 1

        self.hora_llegada = env.now
        self.hora_salida = None

        self.estado = None  # "en_cola", "siendo_atendido", "terminado"
        self.sector_actual = None

        self.sectores_a_visitar = []
        self.cantidad_items = 0

        self.nivel_prioridad = CLIENTE_ORDEN[tipo]


class Sector:

    def __init__(self, env: simpy.Environment, nombre: str, capacidad_de_cliente: int, capacidad_items: int, balanzas: int = 0):
        self.env = env
        self.nombre = nombre
        self.clientes = simpy.Resource(env, capacity=capacidad_de_cliente)
        self.stock = simpy.Container(
            env, init=capacidad_items, capacity=capacidad_items)
        self.balanzas = simpy.Resource(
            env, capacity=balanzas) if balanzas > 0 else None

    # Cliente espacio (request/release)
    def request_espacio(self):
        return self.clientes.request()

    def release_espacio(self, req: simpy.resources.resource.Request):
        """Libera un request previamente adquirido."""
        self.clientes.release(req)

    # Fila para la balanza (si aplica)
    def sacar_items(self, cantidad: int):
        cantidad_hay = self.cuanto_stock()
        if cantidad > cantidad_hay:
            # Devuelve lo que hay, aunque no sea suficiente
            return self.stock.get(cantidad_hay)
        return self.stock.get(cantidad)

    def reponer_items(self, cantidad: int):
        """Devuelve el evento para reponer ítems al stock."""
        return self.stock.put(cantidad)

    def cuanto_stock(self) -> int:
        return int(self.stock.level)

    # Balanzas (si aplica)
    def request_balanza(self):
        if self.balanzas is None:
            raise RuntimeError(f"Sector {self.name} no tiene balanzas")
        return self.balanzas.request()


class Almacen(Sector):
    def __init__(self, env: simpy.Environment, capacidad_de_cliente: int = CAPACIDAD_SECTOR["almacen"],
                 capacidad_items: int = CAPACIDAD_SECTOR_ITEM["almacen"], balanzas: int = 0):
        super().__init__(env, "almacen", capacidad_de_cliente, capacidad_items, balanzas=0)


class Verduleria(Sector):
    def __init__(self, env: simpy.Environment, capacidad_de_cliente: int = CAPACIDAD_SECTOR["verduleria"],
                 capacidad_items: int = CAPACIDAD_SECTOR_ITEM["verduleria"], balanzas: int = BALANZAS_POR_SECTOR["verduleria"]):
        super().__init__(env, "verduleria", capacidad_de_cliente,
                         capacidad_items, balanzas=balanzas)


class Panaderia(Sector):
    def __init__(self, env: simpy.Environment, capacidad_de_cliente: int = CAPACIDAD_SECTOR["panaderia"],
                 capacidad_items: int = CAPACIDAD_SECTOR_ITEM["panaderia"], balanzas: int = BALANZAS_POR_SECTOR["panaderia"]):
        super().__init__(env, "panaderia", capacidad_de_cliente,
                         capacidad_items, balanzas=balanzas)


class Refrigerados(Sector):
    def __init__(self, env: simpy.Environment, capacidad_de_cliente: int = CAPACIDAD_SECTOR["refrigerados"],
                 capacidad_items: int = CAPACIDAD_SECTOR_ITEM["refrigerados"], balanzas: int = 0):
        super().__init__(env, "refrigerados", capacidad_de_cliente, capacidad_items, balanzas=0)


class Supermercado:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.capacidad_maxima = C_MAX
        self.clientes_en_tienda = 0

        self.caja_auto = simpy.Resource(env, capacity=CAJAS_POR_TIPO["auto"])
        self.cajas_preferenciales = [simpy.PriorityResource(env, capacity=1)
                                     for _ in range(CAJAS_POR_TIPO["preferencial"])]
        self.cajas_normales = [simpy.PriorityResource(env, capacity=1)
                               for _ in range(CAJAS_POR_TIPO["normal"])]

    def entrar(self, cliente: Cliente) -> bool:
        if self.clientes_en_tienda >= self.capacidad_maxima:
            cliente.estado = "rechazado"
            return False
        self.clientes_en_tienda += 1
        return True

    def salir(self, cliente: Cliente):
        if self.clientes_en_tienda > 0:
            cliente.estado = "terminado"
            self.clientes_en_tienda -= 1

    def request_caja_auto(self):
        return self.caja_auto.request()

    def request_caja_preferencial(self, idx: int, cliente: Cliente):
        return self.cajas_preferenciales[idx].request(priority=cliente.nivel_prioridad)

    def request_caja_normal(self, idx: int, cliente: Cliente):
        return self.cajas_normales[idx].request(priority=cliente.nivel_prioridad)

    def _cola_len(self, recurso: simpy.resources.resource.Resource) -> int:
        return len(recurso.queue) + getattr(recurso, 'count', 0)

    def elegir_caja(self, cliente: Cliente, rng: np.random.Generator):
        candidatos = []

        if cliente.cantidad_items <= LIMITE_CAJA_AUTO:
            if cliente.tipo == "P":
                candidatos.append(
                    ("auto", None, self._cola_len(self.caja_auto) / 2))
            else:
                candidatos.append(
                    ("auto", None, self._cola_len(self.caja_auto) / 3))

        for i, recurso in enumerate(self.cajas_preferenciales):
            candidatos.append(
                ("preferencial", i, float(self._cola_len(recurso))))

        # cajas normales
        for i, recurso in enumerate(self.cajas_normales):
            candidatos.append(("normal", i, float(self._cola_len(recurso))))

        # Ordenar por el valor contado y quedarse con los de menor valor
        candidatos.sort(key=lambda x: x[2])
        min_cola = candidatos[0][2]
        mejores = [c for c in candidatos if c[2] == min_cola]

        # Desempates explícitos según tipo de cliente
        if cliente.tipo == "P":
            if len(mejores) == 1:
                return mejores[0]
            else:
                cajas_normales = []
                for caja in mejores:
                    if caja[0] == "preferencial":
                        return caja
                    elif caja[0] == "normal":
                        cajas_normales.append(caja)
                if len(cajas_normales) > 0:
                    idx = int(rng.integers(0, len(cajas_normales)))
                    return cajas_normales[idx]
                else:
                    return mejores[0]
        else:
            if len(mejores) == 1:
                return mejores[0]
            else:
                cajas_normales = []
                for caja in mejores:
                    if caja[0] == "normal":
                        cajas_normales.append(caja)
                    elif caja[0] == "auto":
                        return caja
                if len(cajas_normales) > 0:
                    idx = int(rng.integers(0, len(cajas_normales)))
                    return cajas_normales[idx]
                else:
                    return mejores[0]
