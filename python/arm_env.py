"""
arm_env.py  -  '관절 3개짜리 로봇 팔로 목표 지점에 손끝 닿기' 강화학습 환경 (Python 쪽, Gymnasium 규격)

역할 나누기
  Unity  : ArticulationBody 로 관절 물리(중력, 관성, 스프링-댐퍼 구동)를 계산 -> "일어난 일(raw)"만 알려 준다
  Python : raw 로부터 '관측(상태)'을 만들고, '보상'을 계산하고, '종료'를 판단한다   <- 이번 과제의 설계 부분

과제
  바닥에 고정된 팔(관절 3개: 좌우 회전, 어깨, 팔꿈치)의 손끝을 판마다 무작위로 나타나는 목표 지점에 닿게 한다.

Gymnasium 규격
  obs, info = env.reset(seed=None)
  obs, reward, terminated, truncated, info = env.step(action)
    action     : 연속 3차원 (-1 ~ 1).  각 관절의 '목표 각도'를 한 스텝에 얼마나 돌릴지
    terminated : 손끝이 목표에 닿음 (성공),   truncated : 시간 초과(timeout)
"""
import json
import socket

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HOST = "127.0.0.1"
PORT = 9999

# =====================================================================
# [설계 1] 환경 상수  (Unity ArmEnv 의 Inspector 값과 맞춰야 한다)
# =====================================================================
LIMITS_DEG = (150.0, 100.0, 120.0)      # 관절 회전 범위 (±)
SHOULDER = np.array([0.0, 0.6, 0.0])    # 어깨(관절1) 위치
REACH = 2.6                             # 팔을 완전히 폈을 때 어깨에서 손끝까지 (1.4 + 1.2)
SUCCESS_DIST = 0.2                      # 손끝-목표 거리가 이보다 작으면 성공
MAX_STEPS = 200                         # 한 스텝 0.1초 -> 20초.  넘으면 timeout
MAX_VEL = 6.0                           # 관절 각속도 정규화 기준 (라디안/초)

# =====================================================================
# [설계 2] 보상 정책
# =====================================================================
R_SUCCESS = 10.0          # 목표에 닿으면 (종료)
K_SHAPE = 3.0             # 접근 보상 계수: K * (이전 거리 - 현재 거리)
R_TIME = -0.01            # 매 스텝 작은 감점 (빨리 닿도록)
R_ENERGY = 0.02           # 행동이 클수록 감점: R_ENERGY * |a|^2   (덜덜 떨거나 과격하게 움직이는 것을 막음)
R_FLOOR = 0.2             # 손끝이 바닥에 너무 가까우면(y < 0.12) 스텝마다 감점
FLOOR_Y = 0.12

# =====================================================================
# [설계 3] 상태(관측) 15차원, 대부분 -1 ~ 1 로 정규화 (손끝→목표 벡터만 최대 ±2)
#   관절 각도 3, 관절 각속도 3, 손끝 위치(어깨 기준) 3, 목표 위치(어깨 기준) 3, 손끝에서 목표로 가는 벡터 3
#   '전지적 시점': Unity 가 아는 정확한 값을 준다.  (실험 novel 로 속도 정보의 중요도를 비교한다)
# =====================================================================
STATE_NAMES = (["q0", "q1", "q2", "w0", "w1", "w2"] + ["tipx", "tipy", "tipz"]
               + ["tgx", "tgy", "tgz"] + ["dx", "dy", "dz"])


def distance(raw):
    return float(np.linalg.norm(np.array(raw["tip"]) - np.array(raw["target"])))


def make_observation(raw, use_vel=True):
    q = np.array(raw["joints"], dtype=np.float64) / np.radians(LIMITS_DEG)      # -1 ~ 1
    w = np.clip(np.array(raw["vel"], dtype=np.float64) / MAX_VEL, -1, 1) if use_vel else np.zeros(3)
    tip = (np.array(raw["tip"]) - SHOULDER) / REACH
    tg = (np.array(raw["target"]) - SHOULDER) / REACH
    rel = (np.array(raw["target"]) - np.array(raw["tip"])) / REACH
    return np.clip(np.concatenate([q, w, tip, tg, rel]), -2.0, 2.0).astype(np.float32)


def compute_reward(prev_dist, raw, action, k_shape=K_SHAPE, r_energy=R_ENERGY):
    """한 스텝의 보상, 현재 거리, 종료 결과.  outcome: 'reach' / None(계속)"""
    dist = distance(raw)
    a = np.asarray(action, dtype=np.float64)
    reward = k_shape * (prev_dist - dist) + R_TIME - r_energy * float(np.sum(a * a))
    if raw["tip"][1] < FLOOR_Y:
        reward -= R_FLOOR
    outcome = None
    if dist < SUCCESS_DIST:
        reward += R_SUCCESS
        outcome = "reach"
    return reward, dist, outcome


class UnityArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host=HOST, port=PORT, timeout=10.0, use_vel=True, k_shape=K_SHAPE, r_energy=R_ENERGY):
        super().__init__()
        self.use_vel = use_vel          # 실험용: False 면 관절 속도를 숨김
        self.k_shape = k_shape          # 실험용: 0 이면 접근 보상 없음
        self.r_energy = r_energy        # 실험용: 0 이면 에너지 감점 없음
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(len(STATE_NAMES),), dtype=np.float32)   # 손끝→목표 벡터는 최대 2 까지 가능
        self.steps = 0
        self.prev_dist = 0.0
        self.floor_steps = 0
        self._needs_reset = True

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print(f"[Python] 대기 중... (포트 {port})  지금 Unity 에서 Play(▶) 버튼을 누르세요.")
        self.conn, addr = self.server.accept()
        self.conn.settimeout(timeout)
        self.rfile = self.conn.makefile("r", encoding="utf-8", newline="\n")
        print(f"[Python] Unity 연결 성공: {addr}")

    def _send(self, obj):
        self.conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def _recv(self):
        try:
            line = self.rfile.readline()
        except socket.timeout:
            raise RuntimeError("Unity 가 10초 동안 응답하지 않았습니다. Unity 의 Console 창을 확인하세요.")
        if not line:
            raise ConnectionError("Unity 연결이 끊어졌습니다. (Play 를 멈췄나요?)")
        return json.loads(line)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        unity_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._send({"cmd": "reset", "seed": unity_seed})
        raw = self._recv()
        self.steps = 0
        self.floor_steps = 0
        self.prev_dist = distance(raw)
        self._needs_reset = False
        return make_observation(raw, self.use_vel), {"dist": self.prev_dist}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("에피소드가 끝났습니다. step() 전에 reset() 을 먼저 호출하세요.")
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        self._send({"cmd": "step", "a0": float(a[0]), "a1": float(a[1]), "a2": float(a[2])})
        raw = self._recv()
        self.steps += 1

        reward, dist, outcome = compute_reward(self.prev_dist, raw, a, self.k_shape, self.r_energy)
        self.prev_dist = dist
        if raw["tip"][1] < FLOOR_Y:
            self.floor_steps += 1

        terminated = outcome is not None
        truncated = (not terminated) and self.steps >= MAX_STEPS
        if truncated:
            outcome = "timeout"
        self._needs_reset = terminated or truncated

        info = {"outcome": outcome, "dist": dist, "floor_steps": self.floor_steps}
        return make_observation(raw, self.use_vel), float(reward), terminated, truncated, info

    def close(self):
        for s in (getattr(self, "rfile", None), getattr(self, "conn", None), getattr(self, "server", None)):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
