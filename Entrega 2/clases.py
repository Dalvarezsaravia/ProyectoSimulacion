import simpy
import numpy as np
from parametros import (
    CAPACIDAD_SECTOR,
    CAPACIDAD_SECTOR_ITEM,
    BALANZAS_POR_SECTOR,
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
        """Devuelve un evento request para usar en `with sector.request_espacio(): yield req`"""
        return self.clientes.request()

    def release_espacio(self, req: simpy.resources.resource.Request):
        """Libera un request previamente adquirido."""
        self.clientes.release(req)

    # Fila para la balanza (si aplica)
    def sacar_items(self, amount: int):
        """Devuelve el evento para tomar ítems del stock."""
        return self.stock.get(amount)

    def reponer_items(self, amount: int):
        """Devuelve el evento para reponer ítems al stock."""
        return self.stock.put(amount)

    def cuanto_stock(self) -> int:
        return int(self.stock.level)

    # Balanzas (si aplica)
    def request_balanza(self):
        if self.balanzas is None:
            raise RuntimeError(f"Sector {self.name} no tiene balanzas")
        return self.balanzas.request()

    def salir_balanza(self, req: simpy.resources.resource.Request):
        if self.balanzas is None:
            raise RuntimeError(f"Sector {self.name} no tiene balanzas")
        self.balanzas.release(req)


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
