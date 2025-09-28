import logging
import os
import time

import numpy as np
from gymnasium import spaces
from alpyne.sim import AnyLogicSim
from alpyne.constants import EngineState

OBS_QUEUES   = "queues"
OBS_PHASES   = "phaseIdx"
ACTION_FIELD = "sw"

PAUSE_WAIT_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.05

log = logging.getLogger("traffic_alpyne_env")
logging.basicConfig(level=logging.INFO)

class TrafficAlpyneEnv:
    metadata = {"render.modes": []}

    def __init__(self, model_path: str):
        java_exe = os.environ.get("JAVA_EXE", None)

        self.sim = AnyLogicSim(
            model_path=model_path,
            java_exe=java_exe,
            auto_lock=True,
            auto_finish=False,
            py_log_level=True,
            java_log_level=True
        )

        schema = self.sim.schema
        obs_keys = list(schema.observation.keys())
        act_keys = list(schema.action.keys())
        log.info("RL schema | observation keys = %s", obs_keys)
        log.info("RL schema | action keys      = %s", act_keys)

        st = self._wait_for_pause_or_terminal(self.sim.reset())

        vec0, n_queues, n_lights = self._obs_to_vec_and_sizes(st.observation)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=vec0.shape, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * n_lights)

        self.n_approaches = n_queues
        self.n_lights = n_lights

        self._last_status = st

    def _wait_for_pause_or_terminal(self, initial_status):
        st = initial_status
        deadline = time.time() + PAUSE_WAIT_TIMEOUT_S

        while True:
            if st.state in (EngineState.PAUSED, EngineState.FINISHED, EngineState.ERROR):
                return st
            if time.time() > deadline:
                return st
            time.sleep(POLL_INTERVAL_S)
            st = self.sim.status()

    def _obs_to_vec_and_sizes(self, obs_dict):
        queues = np.asarray(list(obs_dict[OBS_QUEUES]), dtype=np.float32)
        phases = np.asarray(list(obs_dict[OBS_PHASES]), dtype=np.float32)
        vec = np.concatenate([queues, phases], axis=0)
        return vec, int(queues.size), int(phases.size)

    def reset(self, seed = None, options = None):
        st = self._wait_for_pause_or_terminal(self.sim.reset())
        self._last_status = st

        vec, _, _ = self._obs_to_vec_and_sizes(st.observation)
        info = {"engine_state": st.state.name, "message": st.message}
        return vec, info

    def step(self, action):
        st = self._wait_for_pause_or_terminal(self.sim.status())

        if isinstance(action, (int, np.integer)):
            action = [int(action)] * self.n_lights
        else:
            action = list(map(int, np.asarray(action).ravel().tolist()))

        if len(action) < self.n_lights:
            action = action + [0] * (self.n_lights - len(action))
        elif len(action) > self.n_lights:
            action = action[: self.n_lights]

        st = self.sim.take_action(**{ACTION_FIELD: action})
        st = self._wait_for_pause_or_terminal(st)
        self._last_status = st

        vec, n_queues, _ = self._obs_to_vec_and_sizes(st.observation)
        reward = -float(vec[:n_queues].sum())

        done = (st.state == EngineState.FINISHED) or bool(st.stop)
        trunc = False
        info = {"engine_state": st.state.name, "message": st.message}
        return vec, reward, done, trunc, info

    def close(self):
        try:
            self.sim.status()
        except Exception:
            pass
