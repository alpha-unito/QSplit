from streamflow.ext.plugin import StreamFlowPlugin

from .connector.quantum_connector import QuantumConnectorWrapper


class QSplitStreamFlowPlugin(StreamFlowPlugin):
    def register(self) -> None:
        self.register_connector("qsplit.quantum_connector", QuantumConnectorWrapper)
