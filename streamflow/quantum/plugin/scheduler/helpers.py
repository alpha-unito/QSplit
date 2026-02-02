from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, MutableMapping
from streamflow.core.scheduling import Hardware
from streamflow.core.workflow import Job
from streamflow.scheduling.scheduler import JobContext, _get_connector_stack


@dataclass(frozen=True)
class TargetState:
    target: Any
    deployment: str
    connector: Any
    available_locations: MutableMapping[str, Any]


async def wait_for_targets(scheduler, targets) -> None:
    if scheduler.retry_interval is not None:
        await asyncio.sleep(scheduler.retry_interval)
        return
    async def _wait(cond):
        async with cond:
            await cond.wait()
    tasks = [
        asyncio.create_task(_wait(scheduler.wait_queues[target.deployment.name]))
        for target in targets
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def load_target_state(scheduler, target) -> TargetState:
    deployment = target.deployment.name
    connector = scheduler.context.deployment_manager.get_connector(deployment)
    available_locations = dict(await connector.get_available_locations(service=target.service))
    return TargetState(
        target=target,
        deployment=deployment,
        connector=connector,
        available_locations=available_locations,
    )


async def load_target_states(scheduler, targets: list[Any]) -> list[TargetState]:
    return list(await asyncio.gather(
        *(asyncio.create_task(load_target_state(scheduler, t)) for t in targets)
    ))


def collect_connectors(target_states: list[TargetState]) -> dict[str, Any]:
    connectors: dict[str, Any] = {}
    for state in target_states:
        for conn in _get_connector_stack(state.connector):
            connectors[conn.deployment_name] = conn
    return connectors


async def lock_connectors(scheduler, connectors: dict[str, Any], exit_stack) -> None:
    for deployment_name in sorted(connectors):
        await exit_stack.enter_async_context(
            scheduler.locks.setdefault(deployment_name, asyncio.Lock())
        )


async def build_candidates(
    scheduler,
    target_states: list[TargetState],
    job_context: JobContext,
    hardware_requirement,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    candidate_locations: dict[str, Any] = {}
    candidate_meta: dict[int, dict[str, Any]] = {}
    for state in target_states:
        target = state.target
        deployment = state.deployment
        connector = state.connector
        available_locations = state.available_locations
        if not available_locations:
            continue
        job_instance = Job(
            name=job_context.job.name,
            workflow_id=job_context.job.workflow_id,
            inputs=job_context.job.inputs,
            input_directory=job_context.job.input_directory or target.workdir,
            output_directory=job_context.job.output_directory or target.workdir,
            tmp_directory=job_context.job.tmp_directory or target.workdir,
        )
        job_hardware = hardware_requirement.eval(job_instance) if hardware_requirement else Hardware()
        hardware_requirements: dict[str, Any] = {}
        for requirements in await asyncio.gather(
            *(
                asyncio.create_task(scheduler._resolve_hardware_requirement(connector, location, job_hardware))
                for location in available_locations.values()
            )
        ):
            for key, hardware in requirements.items():
                if key not in hardware_requirements:
                    hardware_requirements[key] = hardware
                else:
                    hardware_requirements[key] |= hardware
        valid_locations = {
            k: loc
            for k, loc in available_locations.items()
            if scheduler._is_valid(
                connector=connector,
                location=loc,
                hardware_requirements=hardware_requirements,
                job_name=job_instance.name,
            )
        }
        if len(valid_locations) < target.locations:
            continue
        for loc_name, loc in valid_locations.items():
            key = loc_name
            if key in candidate_locations:
                key = f"{deployment}:{loc_name}"
            candidate_locations[key] = loc
            candidate_meta[id(loc)] = {
                "target": target,
                "connector": connector,
                "hardware": hardware_requirements,
            }
    return candidate_locations, candidate_meta


async def allocate_candidate(
    scheduler,
    policy,
    job_context: JobContext,
    candidate_locations: dict[str, Any],
    candidate_meta: dict[int, dict[str, Any]],
) -> bool:
    selected_location = await policy.get_location(
        context=scheduler.context,
        job=job_context.job,
        hardware_requirement=Hardware(),
        available_locations=candidate_locations,
        jobs=scheduler.job_allocations,
        locations=scheduler.location_allocations,
    )
    if selected_location is None:
        return False
    meta = candidate_meta.get(id(selected_location))
    if meta is None:
        for loc in candidate_locations.values():
            if loc == selected_location:
                meta = candidate_meta.get(id(loc))
                break
    if meta is None:
        return False
    scheduler._allocate_job(
        job=job_context.job,
        hardware=meta["hardware"],
        connector=meta["connector"],
        selected_locations=[selected_location],
        target=meta["target"],
    )
    job_context.scheduled = True
    return True
