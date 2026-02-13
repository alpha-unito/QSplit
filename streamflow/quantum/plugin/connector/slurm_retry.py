from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable


def contains_slurm_cancellation(output: str) -> bool:
    upper = output.upper()
    return "*** JOB" in upper and "CANCELLED AT" in upper and "DUE TO" in upper


async def run_with_slurm_cancellation_retry(
    *,
    logger: logging.Logger,
    service_candidates: list[str],
    get_location_for_service: Callable[[int, str], Awaitable[object | None]],
    run_on_location: Callable[[object], Awaitable[tuple[str, int] | tuple[bytes, int] | None]],
    retries: int = 2,
    retry_delay: int = 2,
) -> tuple[str, int] | None:
    last_cancel_output = ""
    last_cancel_code = 143
    for service_index, target_service in enumerate(service_candidates):
        attempt = 0
        while True:
            target_location = await get_location_for_service(service_index, target_service)
            if target_location is None:
                break
            result = await run_on_location(target_location)
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                output, return_code = result
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                if isinstance(output, str) and contains_slurm_cancellation(output):
                    last_cancel_output = output
                    if isinstance(return_code, int) and return_code != 0:
                        last_cancel_code = return_code
                    if attempt < retries:
                        attempt += 1
                        logger.warning(
                            "Detected Slurm cancellation on service '%s', retrying (%s/%s): %s",
                            target_service,
                            attempt,
                            retries,
                            output,
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    break
                return output, return_code
            return result
        if service_index < len(service_candidates) - 1:
            logger.warning(
                "Switching service from '%s' to fallback '%s' after repeated Slurm cancellations.",
                target_service,
                service_candidates[service_index + 1],
            )
    if last_cancel_output:
        logger.warning(
            "Detected Slurm cancellation in connector output, forcing non-zero exit code: %s",
            last_cancel_output,
        )
        return last_cancel_output, last_cancel_code
    return "", 143
