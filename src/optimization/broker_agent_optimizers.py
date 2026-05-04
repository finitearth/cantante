import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def broker_agent_optimizer_step(optim):
    """
    Externalized version of the graph-level optimization step.
    Handles concurrency and lifecycle, while delegating the optimization logic
    (select → evaluate → attribute → log) to `optim.select_evaluate_attribute`.
    """

    logger.info("Stepping agent optimizers...")

    executor = ThreadPoolExecutor(thread_name_prefix="agent")
    futures = [executor.submit(optim._step) for optim in optim.agent_optimizers.values()]

    def collect_pending():
        out = {}
        for agent_name, proxy in optim.proxy_tasks.items():
            p = proxy.try_pop_pending()
            if p is not None:
                out[agent_name] = p
        return out

    try:
        while True:
            # -----------------------------------------------------
            # Crash detection
            # -----------------------------------------------------
            for f in futures:
                if f.done():
                    exc = f.exception()
                    if exc is not None:
                        logger.error("Worker crashed", exc_info=exc)
                        for p in optim.proxy_tasks.values():
                            p.cancel_pending()
                        raise exc

            # -----------------------------------------------------
            # Pending work available?
            # -----------------------------------------------------
            pending_by_agent = collect_pending()
            if pending_by_agent:
                try:
                    # Core optimization pipeline
                    attributed_scores = optim.select_evaluate_attribute(pending_by_agent)

                    # Fulfill futures - only return scores for each agent's actual prompts
                    for agent_name, pending in pending_by_agent.items():
                        pending.scores = attributed_scores[agent_name][: len(pending.prompts)]
                        pending.done.set()

                except Exception as e:
                    logger.error("Broker error", exc_info=e)
                    for p in pending_by_agent.values():
                        p.error = e
                        p.done.set()
                    raise

            # -----------------------------------------------------
            # Check if completely done
            # -----------------------------------------------------
            if all(f.done() for f in futures) and not collect_pending():
                break

            time.sleep(0.005)  # avoid CPU spin

    except Exception:
        logger.error("Aborting step", exc_info=True)

        for f in futures:
            f.cancel()

        for p in optim.proxy_tasks.values():
            p.cancel_pending()

        executor.shutdown(wait=False, cancel_futures=True)
        raise

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
