"""
pick_env.py  -  '집게로 물체를 집어 같은 색 상자에 넣기' 강화학습 환경 (Python 쪽, Gymnasium 규격)

도달 과제(arm_env.py)의 팔·통신을 그대로 쓰고, 집게와 물체·상자를 더했다.
  Unity  : 물체가 손끝에 붙었는지(held), 떨어진 물체의 낙하·충돌, 물체가 어느 상자 안에 있는지(in_box)를 알려 준다
  Python : 관측·보상·종료를 정한다   <- 설계 부분

과제
  바닥 위 물체(빨강 또는 파랑)를 집어서, 같은 색 상자 위로 옮긴 뒤 놓는다.
  상자 위치는 고정 (빨강 +Z 쪽, 파랑 -Z 쪽).  물체 위치와 색은 판마다 무작위.

Gymnasium 규격
  action     : 연속 4차원 (-1 ~ 1).  관절 3개의 목표 각도 변화량 + 집게(>0 닫기, <=0 열기)
  terminated : 물체가 맞는 상자 안(place, 성공) / 틀린 상자 안(wrong) / 물체를 잃어버림(lost)
  truncated  : 시간 초과(timeout)
"""
import numpy as np
from gymnasium import spaces

from arm_env import FLOOR_Y, LIMITS_DEG, MAX_VEL, R_FLOOR, R_TIME, REACH, SHOULDER, UnityArmEnv

# =====================================================================
# [설계 1] 환경 상수  (Unity ArmEnv 의 Pick 항목과 맞춰야 한다)
# =====================================================================
MAX_STEPS = 300                         # 30초
DROP_HEIGHT = 0.7                       # 상자 바닥 중심 위 이 높이를 '놓을 지점'으로 삼는다

# =====================================================================
# [설계 2] 보상 정책
#   단계마다 거리를 줄이면 보상(shaping).  단계가 바뀌어도 값이 튀지 않도록 '잠재 함수(potential)' 하나로 정의한다.
#     잡기 전 : phi = -(손끝~물체 거리 + 물체~놓을 지점 거리 + C_HOLD)
#     잡은 뒤 : phi = -(물체~놓을 지점 거리)
#     성공    : phi = 0
#   보상 = K_SHAPE * (phi_현재 - phi_이전)
#   -> 잡으면 +K*C_HOLD, 엉뚱한 곳에 놓으면 그만큼 다시 깎인다.  들었다 놨다 반복으로 점수를 벌 수 없다.
# =====================================================================
R_SUCCESS = 10.0          # 맞는 상자에 넣으면 (종료)
R_WRONG = 5.0             # 틀린 상자에 넣으면 감점 (종료)
R_LOST = 5.0              # 물체가 바닥 밖으로 떨어지는 등 잃어버리면 감점 (종료)
K_SHAPE = 3.0
C_HOLD = 1.0              # '잡고 있음'의 가치 (거리 단위)
R_ENERGY = 0.02           # 관절 행동 크기 감점 (집게 행동은 제외)
R_CENTER = 0.0            # (v2) 성공할 때 상자 가운데에 가까울수록 추가 보상: R_CENTER * (1 - 가운데까지 거리 / BOX_INNER)
BOX_INNER = 0.35          # Unity Box Inner (상자 안쪽 반 너비)

# =====================================================================
# [설계 3] 상태(관측) 21차원
#   관절 각도 3, 관절 각속도 3, 손끝(어깨 기준) 3, 물체(어깨 기준) 3, 손끝->물체 3, 물체->놓을 지점 3,
#   잡고 있음 1, 집게 닫힘 1, 물체 색(빨강 -1 / 파랑 +1) 1
# =====================================================================
STATE_NAMES = (["q0", "q1", "q2", "w0", "w1", "w2", "tipx", "tipy", "tipz", "objx", "objy", "objz",
                "tox", "toy", "toz", "ogx", "ogy", "ogz", "held", "grip", "color"])


def goal_point(raw):
    """물체 색과 같은 상자의 '놓을 지점'"""
    b = np.array(raw["boxes"], dtype=np.float64).reshape(2, 3)[raw["color"]]
    return b + np.array([0.0, DROP_HEIGHT, 0.0])


def potential(raw):
    tip, obj, goal = np.array(raw["tip"]), np.array(raw["obj"]), goal_point(raw)
    d_og = float(np.linalg.norm(goal - obj))
    if raw["held"]:
        return -d_og
    return -(float(np.linalg.norm(obj - tip)) + d_og + C_HOLD)


def make_observation(raw, use_vel=True):
    q = np.array(raw["joints"], dtype=np.float64) / np.radians(LIMITS_DEG)
    w = np.clip(np.array(raw["vel"], dtype=np.float64) / MAX_VEL, -1, 1) if use_vel else np.zeros(3)
    tip, obj, goal = np.array(raw["tip"]), np.array(raw["obj"]), goal_point(raw)
    flags = [float(raw["held"]), float(raw["grip"]), 1.0 if raw["color"] == 1 else -1.0]
    return np.clip(np.concatenate([q, w, (tip - SHOULDER) / REACH, (obj - SHOULDER) / REACH,
                                   (obj - tip) / REACH, (goal - obj) / REACH, flags]), -2.0, 2.0).astype(np.float32)


def judge(raw):
    """물체 상태로 종료 결과를 정한다.  'place' / 'wrong' / 'lost' / None(계속)"""
    if raw["in_box"] >= 0:
        return "place" if raw["in_box"] == raw["color"] else "wrong"
    obj = np.array(raw["obj"])
    if obj[1] < -0.5 or np.linalg.norm(obj[[0, 2]]) > 4.0:
        return "lost"
    return None


def center_error(raw):
    """물체와 목표 상자 가운데 사이의 가로 거리"""
    b = np.array(raw["boxes"], dtype=np.float64).reshape(2, 3)[raw["color"]]
    return float(np.linalg.norm((np.array(raw["obj"]) - b)[[0, 2]]))


def compute_reward(prev_phi, raw, action, k_shape=K_SHAPE, r_energy=R_ENERGY, r_center=R_CENTER):
    """한 스텝의 보상, 현재 잠재값, 종료 결과"""
    outcome = judge(raw)
    phi = 0.0 if outcome == "place" else potential(raw)
    a = np.asarray(action[:3], dtype=np.float64)
    reward = k_shape * (phi - prev_phi) + R_TIME - r_energy * float(np.sum(a * a))
    if raw["tip"][1] < FLOOR_Y:
        reward -= R_FLOOR
    if outcome == "place":
        reward += R_SUCCESS + r_center * max(0.0, 1.0 - center_error(raw) / BOX_INNER)
    elif outcome == "wrong":
        reward -= R_WRONG
    elif outcome == "lost":
        reward -= R_LOST
    return reward, phi, outcome


class UnityPickEnv(UnityArmEnv):
    def __init__(self, use_vel=True, k_shape=K_SHAPE, r_energy=R_ENERGY, task="pick",
                 r_center=R_CENTER, max_delta=None, grab_dist=None, rand_pose=False, **kw):
        """task="conveyor" 면 Unity 가 다음 물체가 컨베이어 끝에 올 때까지 기다렸다가 판을 시작한다 (시연용)
        max_delta / grab_dist / rand_pose 는 reset 때 Unity 로 보내는 v2 설정 (None/False 면 Unity Inspector 값)"""
        if task == "conveyor":
            kw.setdefault("timeout", 120.0)      # 다음 물체를 기다리는 동안 응답이 늦다
        super().__init__(use_vel=use_vel, k_shape=k_shape, r_energy=r_energy, **kw)
        self.task = task
        self.r_center = r_center
        self.unity_opts = {}
        if max_delta:
            self.unity_opts["max_delta"] = float(max_delta)
        if grab_dist:
            self.unity_opts["grab_dist"] = float(grab_dist)
        if rand_pose:
            self.unity_opts["rand_pose"] = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(len(STATE_NAMES),), dtype=np.float32)
        self.prev_phi = 0.0
        self.grabbed = False
        self.raw = None

    def reset(self, *, seed=None, options=None):
        super(UnityArmEnv, self).reset(seed=seed)
        unity_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._send({"cmd": "reset", "seed": unity_seed, "task": self.task, **self.unity_opts})
        raw = self._recv()
        self.raw = raw
        self.steps = 0
        self.floor_steps = 0
        self.grabbed = False
        self.prev_phi = potential(raw)
        self._needs_reset = False
        return make_observation(raw, self.use_vel), {"color": raw["color"]}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("에피소드가 끝났습니다. step() 전에 reset() 을 먼저 호출하세요.")
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)
        self._send({"cmd": "step", "a0": float(a[0]), "a1": float(a[1]), "a2": float(a[2]), "a3": float(a[3])})
        raw = self._recv()
        self.raw = raw
        self.steps += 1

        reward, phi, outcome = compute_reward(self.prev_phi, raw, a, self.k_shape, self.r_energy, self.r_center)
        self.prev_phi = phi
        self.grabbed = self.grabbed or bool(raw["held"])
        if raw["tip"][1] < FLOOR_Y:
            self.floor_steps += 1

        terminated = outcome is not None
        truncated = (not terminated) and self.steps >= MAX_STEPS
        if truncated:
            outcome = "timeout"
        self._needs_reset = terminated or truncated

        info = {"outcome": outcome, "grabbed": self.grabbed, "held": raw["held"],
                "goal_dist": float(np.linalg.norm(goal_point(raw) - np.array(raw["obj"]))),
                "center_err": center_error(raw),
                "floor_steps": self.floor_steps}
        return make_observation(raw, self.use_vel), float(reward), terminated, truncated, info
