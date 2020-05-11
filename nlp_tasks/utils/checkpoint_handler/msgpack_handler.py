import os
import typing
from enum import Enum
from hashlib import md5

from loguru import logger
from prefect.core.edge import Edge
from prefect.engine.result import Result
from prefect.engine.state import State, Success

from ..task_runner import DSTaskRunner
from .bencode import encode


def msgpack_checkpoint_handler(
    task_runner: DSTaskRunner, old_state: State, new_state: State
) -> State:
    """
    A handler designed to implement result caching by filename. If the result handler's ``read``
    method can be successfully run, this handler loads the result of that method as the task result
    and sets the task state to ``Success``. Similarly, on successful
    completion of the task, if the task was actually run and not loaded from cache, this handler
    will apply the result handler's ``write`` method to the task.
    Parameters
    ----------
    task_runner : instance of DSTaskRunner
        The task runner associated with the flow the handler is used in.
    old_state : instance of prefect.engine.state.State
        The current state of the task.
    new_state : instance of prefect.engine.state.State
        The expected new state of the task.
    Returns
    -------
    new_state : instance of prefect.engine.state.State
        The actual new state of the task.
    """
    if (
        "PREFECT__FLOWS__CHECKPOINTING" in os.environ
        and os.environ["PREFECT__FLOWS__CHECKPOINTING"] == "true"
    ):
        raise AttributeError(
            "Cannot use standard prefect checkpointing with this handler"
        )
    if task_runner.result_handler is not None:
        input_mapping = None
        for edge, state in sorted(
            task_runner.upstream_states.items(), key=lambda item: item[0].key
        ):
            key = edge.key
            try:
                if isinstance(state.result, Enum):
                    hashed_val = md5(encode(state.result.value)).hexdigest()
                else:
                    hashed_val = md5(encode(state.result)).hexdigest()
                input_mapping = f"{key}:{hashed_val}|{input_mapping}"
            except KeyError:
                logger.error(
                    f"Unable to encode argument {key}. Will not be conisdered for caching validation!!!"
                )
    if (
        task_runner.result_handler is not None
        and old_state.is_pending()
        and new_state.is_running()
    ):
        try:
            data = task_runner.task.result_handler.read(input_mapping)
        except FileNotFoundError:
            return new_state
        except TypeError:  # unexpected argument input_mapping
            raise TypeError(
                "Result handler could not accept input_mapping argument. "
                "Please ensure that you are using a handler from prefect_ds."
            )
        result = Result(value=data, result_handler=task_runner.task.result_handler)
        state = Success(result=result, message="Task loaded from disk.")
        return state

    if (
        task_runner.result_handler is not None
        and old_state.is_running()
        and new_state.is_successful()
    ):
        task_runner.task.result_handler.write(new_state.result, input_mapping)

    return new_state
