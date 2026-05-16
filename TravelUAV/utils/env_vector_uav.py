import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Thread
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
import numpy as np
import attr
import copy

from src.common.param import args

from utils.pickle5_multiprocessing import ConnectionWrapper
from utils.env_utils_uav import ENV
from utils.logger import logger


ENV_TIMING_LOG_ENABLED = False


def env_timing_log(message):
    if ENV_TIMING_LOG_ENABLED:
        logger.info(message)


COMMAND_CLOSE = "close"
COMMAND_SET_BATCH = "set_batch"
COMMAND_GET_OBS = "get_obs_at"
COMMAND_GET_COLLISION_SENSOR = 'get_collision_sensor'


try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch
    from torch import multiprocessing as mp  # type:ignore
except ImportError:
    torch = None
    import multiprocessing as mp  # type:ignore


@attr.s(auto_attribs=True, slots=True)
class _ReadWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    should have something to read.
    """
    read_fn: Callable[[], Any]
    rank: int
    is_waiting: bool = False

    def __call__(self) -> Any:
        if not self.is_waiting:
            raise RuntimeError(
                f"Tried to read from process {self.rank}"
                " but there is nothing waiting to be read"
            )
        res = self.read_fn()
        self.is_waiting = False

        return res


@attr.s(auto_attribs=True, slots=True)
class _WriteWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    can be written to safely.  In other words, checks to make sure the
    result returned from the last write was read.
    """
    write_fn: Callable[[Any], None]
    read_wrapper: _ReadWrapper

    def __call__(self, data: Any) -> None:
        if self.read_wrapper.is_waiting:
            raise RuntimeError(
                f"Tried to write to process {self.read_wrapper.rank}"
                " but the last write has not been read"
            )
        self.write_fn(data)
        self.read_wrapper.is_waiting = True


class VectorEnvUtil:

    _num_envs: int
    _mp_ctx: BaseContext
    _workers: List[Union[mp.Process, Thread]]
    _connection_read_fns: List[_ReadWrapper]
    _connection_write_fns: List[_WriteWrapper]

    def __init__(
        self,
        load_scenes,
        num_envs: int = 1,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`env.Env` or :ref:`env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            :ref:`_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        :param workers_ignore_signals: Whether or not workers will ignore SIGINT and SIGTERM
            and instead will only exit when :ref:`close` is called
        """
        self._is_closed = True

        self.load_scenes = load_scenes
        self._num_envs = int(num_envs)

        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args={
                'load_scenes': load_scenes,
            },
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn_args: dict,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = ENV(
            load_scenes=env_fn_args['load_scenes'],
        )

        if parent_pipe is not None:
            parent_pipe.close()

        try:
            command, data = connection_read_fn()
            while command != COMMAND_CLOSE:
                if command == COMMAND_SET_BATCH:
                    env.set_batch(data)
                    connection_write_fn(True)

                elif command == COMMAND_GET_OBS:
                    index, state = data
                    worker_obs_start = time.perf_counter()
                    (teacher_action, done, oracle_success), state = env.get_obs_at(index, state)
                    env_timing_log(
                        f"[TIMING][VectorEnvUtil.worker] env.get_obs_at index={index}: "
                        f"{time.perf_counter() - worker_obs_start:.3f}s"
                    )
                    worker_send_start = time.perf_counter()
                    connection_write_fn(
                        ((teacher_action, done, oracle_success), state)
                    )
                    env_timing_log(
                        f"[TIMING][VectorEnvUtil.worker] send result index={index}: "
                        f"{time.perf_counter() - worker_send_start:.3f}s"
                    )

                elif command == COMMAND_GET_COLLISION_SENSOR:
                    index, state = data
                    is_collision = env.get_collision_sensor_result_at(index, state)
                    connection_write_fn(bool(is_collision))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()
        except KeyboardInterrupt:
            print("Worker KeyboardInterrupt")
        except Exception as e:
            logger.error(e)
            try:
                logger.error('command is: {} \t data is: {}'.format(command, data))
            except:
                pass
        finally:
            if child_pipe is not None:
                child_pipe.close()

    def _spawn_workers(
        self,
        env_fn_args,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        parent_connections, worker_connections = zip(
            *[
                [ConnectionWrapper(c) for c in self._mp_ctx.Pipe(duplex=True)]
                for _ in range(self._num_envs)
            ]
        )
        self._workers = []
        for worker_conn, parent_conn in zip(worker_connections, parent_connections):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    env_fn_args,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()

        read_fns = [
            _ReadWrapper(p.recv, rank)
            for rank, p in enumerate(parent_connections)
        ]
        write_fns = [
            _WriteWrapper(p.send, read_fn)
            for p, read_fn in zip(parent_connections, read_fns)
        ]

        return read_fns, write_fns

    def close(self) -> None:
        if self._is_closed:
            return

        for read_fn in self._connection_read_fns:
            if read_fn.is_waiting:
                read_fn()
                
        for write_fn in self._connection_write_fns:
            write_fn((COMMAND_CLOSE, ''))

        for process in self._workers:
            process.join()

        self._is_closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def set_batch(self, batch):
        set_batch_start = time.perf_counter()
        copy_start = time.perf_counter()
        self.batch = copy.deepcopy(batch)
        env_timing_log(f"[TIMING][VectorEnvUtil.set_batch] deepcopy local batch: {time.perf_counter() - copy_start:.3f}s")

        send_start = time.perf_counter()
        for index in range(self._num_envs):
            worker_copy_start = time.perf_counter()
            batch_for_worker = copy.deepcopy(batch)
            env_timing_log(
                f"[TIMING][VectorEnvUtil.set_batch] deepcopy batch for worker index={index}: "
                f"{time.perf_counter() - worker_copy_start:.3f}s"
            )
            write_start = time.perf_counter()
            self._connection_write_fns[index](
                (COMMAND_SET_BATCH, batch_for_worker)
            )
            env_timing_log(
                f"[TIMING][VectorEnvUtil.set_batch] send batch to worker index={index}: "
                f"{time.perf_counter() - write_start:.3f}s"
            )
        env_timing_log(f"[TIMING][VectorEnvUtil.set_batch] send all workers: {time.perf_counter() - send_start:.3f}s")

        recv_start = time.perf_counter()
        results = [
            self._connection_read_fns[index]() for index in range(self._num_envs)
        ]
        env_timing_log(f"[TIMING][VectorEnvUtil.set_batch] recv all workers: {time.perf_counter() - recv_start:.3f}s")
        env_timing_log(f"[TIMING][VectorEnvUtil.set_batch] total: {time.perf_counter() - set_batch_start:.3f}s")

        return


    def get_obs(self, obs_states) -> Tuple[List[Any], List[Any]]:
        get_obs_start = time.perf_counter()
        self.obs_states = obs_states

        send_all_start = time.perf_counter()
        for index in range(len(obs_states)):
            _, _, state, _, _ = obs_states[index]
            send_start = time.perf_counter()
            self._connection_write_fns[index](
                (COMMAND_GET_OBS, (index, state))
            )
            env_timing_log(
                f"[TIMING][VectorEnvUtil.get_obs] send COMMAND_GET_OBS index={index}: "
                f"{time.perf_counter() - send_start:.3f}s"
            )
        env_timing_log(f"[TIMING][VectorEnvUtil.get_obs] send all: {time.perf_counter() - send_all_start:.3f}s")

        recv_all_start = time.perf_counter()
        results = []
        for index in range(len(obs_states)):
            recv_start = time.perf_counter()
            result = self._connection_read_fns[index]()
            env_timing_log(
                f"[TIMING][VectorEnvUtil.get_obs] recv COMMAND_GET_OBS index={index}: "
                f"{time.perf_counter() - recv_start:.3f}s"
            )
            results.append(result)
        env_timing_log(f"[TIMING][VectorEnvUtil.get_obs] recv all: {time.perf_counter() - recv_all_start:.3f}s")

        format_all_start = time.perf_counter()
        obs = []
        sim_states = []
        for index in range(len(obs_states)):
            format_start = time.perf_counter()
            (teacher_action, done, oracle_success), sim_state = results[index]
            self.obs_states[index] = (obs_states[index][0], obs_states[index][1], sim_state, obs_states[index][3], obs_states[index][4])

            obs.append(
                self._format_obs_at(index, teacher_action, done, oracle_success)
            )
            sim_states.append(sim_state)
            env_timing_log(
                f"[TIMING][VectorEnvUtil.get_obs] format obs index={index}: "
                f"{time.perf_counter() - format_start:.3f}s"
            )
        env_timing_log(f"[TIMING][VectorEnvUtil.get_obs] format all: {time.perf_counter() - format_all_start:.3f}s")
        env_timing_log(f"[TIMING][VectorEnvUtil.get_obs] total: {time.perf_counter() - get_obs_start:.3f}s")

        return obs, sim_states

    def _format_obs_at(self, index: int, teacher_waypoints, done, oracle_success):
        format_start = time.perf_counter()
        rgb_images, depth_images, sim_state, rgb_records, depth_records = self.obs_states[index]
        slice_start = time.perf_counter()
        observations = [info for info in sim_state.trajectory[-5:]]
        env_timing_log(
            f"[TIMING][VectorEnvUtil._format_obs_at] slice trajectory index={index}: "
            f"{time.perf_counter() - slice_start:.3f}s trajectory_len={len(sim_state.trajectory)}"
        )
        assign_start = time.perf_counter()
        observations[-1]['instruction'] = sim_state.raw_trajectory_info['instruction']
        observations[-1]['trajectory_dir'] = sim_state.raw_trajectory_info['trajectory_dir']
        observations[-1]['teacher_action'] = teacher_waypoints
        observations[-1]['rgb'] = rgb_images
        observations[-1]['depth'] = depth_images
        observations[-1]['rgb_record'] = rgb_records
        observations[-1]['depth_record'] = depth_records
        collision = sim_state.is_collisioned
        env_timing_log(
            f"[TIMING][VectorEnvUtil._format_obs_at] assign fields index={index}: "
            f"{time.perf_counter() - assign_start:.3f}s"
        )
        env_timing_log(f"[TIMING][VectorEnvUtil._format_obs_at] total index={index}: {time.perf_counter() - format_start:.3f}s")

        return observations, done, collision, oracle_success


    def get_collision_sensor(self, states) -> List[Any]:
        for index in range(len(states)):
            self._connection_write_fns[index](
                (COMMAND_GET_COLLISION_SENSOR, (index, states[index]))
            )

        results = [
            self._connection_read_fns[index]() for index in range(len(states))
        ]

        return results
