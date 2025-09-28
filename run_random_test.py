import os
import numpy as np
from traffic_alpyne_env import TrafficAlpyneEnv

MODEL_PATH = r"C:\al_model\model.jar"

os.environ.setdefault("JAVA_EXE", r"C:\Program Files\Java\jdk-17\bin\java.exe")

print("CWD:", os.getcwd())
print("MODEL_PATH exists?", os.path.exists(MODEL_PATH))
print("MODEL_PATH abs:", os.path.abspath(MODEL_PATH))

env = TrafficAlpyneEnv(MODEL_PATH)

obs, info = env.reset()
print("obs[:10]:", np.round(obs[:10], 3), "| len:", len(obs))
print("reset info:", info)

for t in range(20):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    print(f"t={t:02d} | a={action} | R={reward:.2f} | done={done} | state={info['engine_state']}")
    if done or trunc:
        break

env.close()
