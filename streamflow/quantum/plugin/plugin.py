from streamflow.ext.plugin import StreamFlowPlugin

from .connector.quantum_connector import QuantumConnectorWrapper
from .policy.quantum_dynamic import QuantumDynamicPolicy
from .scheduler.quantum_dynamic import QuantumDynamicScheduler


class QSplitStreamFlowPlugin(StreamFlowPlugin):

    def register(self) -> None:
        self.register_connector("qsplit.quantum_connector", QuantumConnectorWrapper)
        self.register_policy("qsplit.quantum_dynamic_policy", QuantumDynamicPolicy)
        self.register_scheduler("qsplit.quantum_dynamic_scheduler", QuantumDynamicScheduler)
