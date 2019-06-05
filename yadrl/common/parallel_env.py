import multiprocessing as mp
from multiprocessing import connection
from typing import Any, Dict, Tuple, Union, Sequence

import gym
from numpy import ndarray


class WorkerEnv(mp.Process):
    def __init__(self,
                 worker_id: int,
                 conn: connection.Connection,
                 env_name: str):
        super(WorkerEnv, self).__init__()
        self._worker_id = worker_id
        self._conn = conn
        self._env = gym.make(env_name)

        self._state_machine = {
            'step': self._step,
            'reset': self._reset,
            'close': self._close
        }

    def run(self):
        while True:
            cmd, action = self._conn.recv()
            if self._state_machine[cmd](action):
                break

    def _step(self, action: ndarray):
        state, reward, done, info = self._env.step(action)
        self._conn.send((state, reward, done, info))
        if done:
            self._env.reset()
        return False

    def _reset(self, *args):
        state = self._env.reset()
        self._conn.send(state)
        return False

    def _close(self, *args):
        self._env.close()
        self._conn.send(0)
        print('Worker {} finished job!'.format(self._worker_id))
        return True


class ParallelEnv(object):
    def __init__(self, env_id: str, num_workers: int):
        self._env_id = env_id
        self._num_workers = num_workers
        self._workers, self._controller_conns = self._spawn(num_workers)

    def _spawn(self, num_workers: int):
        result = (self._create_worker(i) for i in range(num_workers))
        workers, controller_conn = zip(*result)
        for worker in workers:
            worker.start()
        return workers, controller_conn

    def _create_worker(self, worker_id: int):
        controller_conn, worker_conn = mp.Pipe(True)
        worker = WorkerEnv(worker_id, worker_conn, self._env_id)
        return worker, controller_conn

    def step(self, actions: Union[ndarray, Sequence[ndarray]]) -> Tuple[
        Tuple[ndarray, ...], Tuple[float, ...],
        Tuple[bool, ...], Tuple[Dict[str, Any], ...]]:
        result = (self._send_and_recv(conn, 'step', actions[i])
                  for i, conn in enumerate(self._controller_conns))
        state, reward, done, info = zip(*result)
        return state, reward, done, info

    def reset(self) -> Tuple[ndarray, ...]:
        states = tuple(self._send_and_recv(conn, 'reset') for conn in
                       self._controller_conns)
        return states

    def close(self):
        for conn in self._controller_conns:
            self._send_and_recv(conn, 'close')
        for worker in self._workers:
            worker.join()

    @staticmethod
    def _send_and_recv(conn, cmd, data=0) -> Any:
        conn.send((cmd, data))
        return conn.recv()
