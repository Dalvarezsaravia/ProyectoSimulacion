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
        self.tiempo_espera_caja = None
        self.hora_salida = None

        self.tiempo_de_permanencia = None

        self.estado = None  # "en_cola", "siendo_atendido", "terminado"
        self.sector_actual = None

        self.sectores_a_visitar = []
        self.cantidad_items = 0
        self.utilidad = 0
        self.tiempo_espera_sector = {"almacen": None, "verduleria": None,
                                     "panaderia": None, "refrigerados": None}
        self.tiempo_espera_balanza = {"verduleria": None, "panaderia": None}

        self.nivel_prioridad = CLIENTE_ORDEN[tipo]


class Sector:

    def __init__(self, env: simpy.Environment, nombre: str, capacidad_de_cliente: int, capacidad_items: int, balanzas: int = 0):
        self.env = env
        self.nombre = nombre
        self.clientes = simpy.Resource(env, capacity=capacidad_de_cliente)
        self.stock = capacidad_items
        self.balanzas = simpy.Resource(
            env, capacity=balanzas) if balanzas > 0 else None
        self.solicitud_de_reponer = False
        self.stock_maximo = capacidad_items

        self.cantidad_de_productos = []

        self.cantidad_clientes_visitados = 0

        self.utilidad_total = 0

    # Cliente espacio (request/release)
    def request_espacio(self):
        return self.clientes.request()

    def release_espacio(self, req: simpy.resources.resource.Request):
        """Libera un request previamente adquirido."""
        self.clientes.release(req)

    # Funcion que permite sacar items del stock del sector, considerando la cantidad deseada y el stock disponible.
    # Si la cantidad deseada es mayor al stock disponible, se ajusta a lo que queda en stock.
    # Luego se actualiza el stock restando la cantidad real sacada y se devuelve esa cantidad real para que el cliente pueda comprarla.
    def sacar_items(self, cantidad: int):

        if self.stock < cantidad:
            cantidad = self.stock
        self.stock -= cantidad
        return cantidad

    def reponer_items(self, cantidad: int):
        """Devuelve el evento para reponer ítems al stock."""

        self.stock += cantidad

    def cuanto_stock(self) -> float:
        return float(self.stock)

    # Balanzas (si aplica)
    def request_balanza(self):
        if self.balanzas is None:
            raise RuntimeError(f"Sector {self.nombre} no tiene balanzas")
        return self.balanzas.request()

# Sectores específicos de la clase Sector, cada uno con sus propias características de capacidad y balanzas (si aplica)


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

        self.caja_auto = simpy.Resource(
            self.env, capacity=CAJAS_POR_TIPO["auto"])
        self.cajas_preferenciales = [simpy.PriorityResource(self.env, capacity=1)
                                     for _ in range(CAJAS_POR_TIPO["preferencial"])]
        self.cajas_normales = [simpy.PriorityResource(self.env, capacity=1)
                               for _ in range(CAJAS_POR_TIPO["normal"])]
        self.reponedores = [Reponedor(self.env), Reponedor(self.env)]

        self.reponedores_disponibles = simpy.Store(
            env, capacity=len(self.reponedores)
        )

        for reponedor in self.reponedores:
            self.reponedores_disponibles.put(reponedor)
        # Para guardar las estadisticas
        self.tiempo_espera_caja_auto = []
        self.tiempo_espera_caja_preferencial = []
        self.tiempo_espera_caja_normal = []

        self.caja_clientes_auto = 0
        self.caja_clientes_preferencial = 0
        self.caja_clientes_normal = 0

        self.clientes_rechazados_por_capacidad = 0

    def entrar(self, cliente: Cliente) -> bool:
        if self.clientes_en_tienda >= self.capacidad_maxima:
            cliente.estado = "rechazado"
            self.clientes_rechazados_por_capacidad += 1
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

    def request_reponedor(self):
        return self.reponedores_disponibles.get()

    def devolver_reponedor(self, reponedor):
        return self.reponedores_disponibles.put(reponedor)

    def _cola_len(self, recurso: simpy.resources.resource.Resource, es_autoservicio: bool = False) -> int:
        # Funcion que devuelve la cantidad de personas en cola para un recurso dado, considerando el caso especial de las cajas de autoservicio
        if es_autoservicio:
            if len(recurso.queue) == 0 and recurso.count < recurso.capacity:
                return 0
            else:
                return len(recurso.queue) + 1
        else:
            return len(recurso.queue) + recurso.count

    def elegir_caja(self, cliente: Cliente, rng: np.random.Generator, registrar_evento):
        candidatos = []
        # Funcion que devuelve una tupla con el tipo de caja, el indice (si aplica) y la cantidad de personas en cola para cada caja disponible,
        # considerando las reglas de prioridad y los tipos de cliente. Luego ordena por la cantidad de personas en cola y aplica desempates según el tipo de cliente.

        if cliente.cantidad_items <= LIMITE_CAJA_AUTO:
            if cliente.tipo == "P":
                candidatos.append(
                    ("auto", None, self._cola_len(self.caja_auto, True) / 2))
            else:
                candidatos.append(
                    ("auto", None, self._cola_len(self.caja_auto, True) / 3))

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
        registrar_evento(
            f"Candidatos del cliente {cliente.id_cliente} ({cliente.tipo}) Cantidad {cliente.cantidad_items}: {candidatos}. Los mejores son: {mejores}")

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
                cajas_preferenciales = []
                for caja in mejores:
                    if caja[0] == "normal":
                        cajas_normales.append(caja)
                    elif caja[0] == "auto":
                        return caja
                    elif caja[0] == "preferencial":
                        cajas_preferenciales.append(caja)
                if len(cajas_normales) > 0:
                    idx = int(rng.integers(0, len(cajas_normales)))
                    return cajas_normales[idx]
                else:
                    return mejores[0]


class Reponedor:
    id = 1

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.id_reponedor = Reponedor.id
        Reponedor.id += 1
        self.estado = "descansando"  # "reponiendo", "descansando"
        self.sector_actual = None
        self.cantidad_items = 0
