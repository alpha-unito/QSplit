from streamflow.ext import StreamFlowPlugin

from .connectors.quantum_availability import QuantumAvailabilityConnector


class QSplitStreamFlowPlugin(StreamFlowPlugin):
    def register(self) -> None:
        # Nome "type" da usare nello streamflow.yml: qsplit.quantum_availability
        # (prefisso qsplit.* per evitare conflitti con core/terze parti)
        self.register_connector("qsplit.quantum_availability", QuantumAvailabilityConnector)