from __future__ import annotations

import asyncio
import contextlib
from importlib.resources import files

from streamflow.core.config import BindingConfig
from streamflow.core.workflow import Job, Status
from streamflow.scheduling.scheduler import DefaultScheduler, JobContext

from .helpers import (
    allocate_candidate,
    build_candidates,
    collect_connectors,
    load_target_states,
    wait_for_targets,
    lock_connectors,
)


class QuantumDynamicScheduler(DefaultScheduler):

    @classmethod
    def get_schema(cls) -> str:
        return (
            files("streamflow.quantum.plugin")
            .joinpath("schemas")
            .joinpath("quantum_dynamic_scheduler.json")
            .read_text("utf-8")
        )

    async def schedule(
        self,
        job: Job,
        binding_config,
        hardware_requirement,
    ) -> None:
        job_context = JobContext(job)
        targets = list(binding_config.targets)
        for f in (self._get_binding_filter(f) for f in binding_config.filters):
            targets = await f.get_targets(job, targets)
        if any(target.locations != 1 for target in targets):
            limited_binding = BindingConfig(targets=targets, filters=[])
            await super().schedule(job, limited_binding, hardware_requirement)
            return
        for target in targets:
            deployment = target.deployment.name
            if deployment not in self.wait_queues:
                self.wait_queues[deployment] = asyncio.Condition()
        policy_config = targets[0].deployment.scheduling_policy
        policy = self._get_policy(policy_config)
        while True:
            async with job_context.lock:
                if job_context.scheduled:
                    return
                target_states = await load_target_states(self, targets)
                connectors = collect_connectors(target_states)
                async with contextlib.AsyncExitStack() as exit_stack:
                    await lock_connectors(self, connectors, exit_stack)
                    candidate_locations, candidate_meta = await build_candidates(
                        self, target_states, job_context, hardware_requirement
                    )
                    if candidate_locations:
                        if await allocate_candidate(self, policy, job_context, candidate_locations, candidate_meta):
                            return
                    else:
                        pass
            await wait_for_targets(self, targets)

    async def notify_status(self, job_name: str, status: Status) -> None:
        await super().notify_status(job_name, status)
