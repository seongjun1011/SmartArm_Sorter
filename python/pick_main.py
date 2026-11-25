"""
pick_main.py  -  집기 환경 실행/검증용 프로그램

  python pick_main.py script         역기구학(IK)으로 계산해서 집고 옮기는 규칙 기반 정책 10판
                                     (학습 없이 이 과제가 풀리는지 = Unity 의 잡기/놓기/상자 판정이 맞는지 확인)
  python pick_main.py random         무작위 행동 20판 (바닥선)
  python pick_main.py check          Gymnasium 규격 검사 (check_env)
  python pick_main.py script 5       5판

반드시 이 프로그램을 먼저 실행한 뒤, Unity 에서 Play(▶)를 누른다.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import itertools
import sys

import numpy as np

from arm_env import LIMITS_DEG, SHOULDER
from pick_env import UnityPickEnv, goal_point

L1, L2 = 1.4, 1.2
MAX_DELTA = np.radians(8.0)          # Unity Max Delta Deg
LIM = np.radians(LIMITS_DEG)


# ---------------------------------------------------------------- 팔 기하 (정기구학 / 역기구학)
def fk(q, s):
    """관절 각도 -> 손끝 위치.  s 는 관절 회전 방향 부호 3개 (Unity 축 방향을 실제 데이터로 맞춘다)"""
    psi, a, b = s[0] * q[0], s[1] * q[1], s[2] * q[2]
    h = np.array([np.cos(psi), 0.0, -np.sin(psi)])
    up = np.array([0.0, 1.0, 0.0])
    return SHOULDER + (L1 * np.sin(a) + L2 * np.sin(a + b)) * h + (L1 * np.cos(a) + L2 * np.cos(a + b)) * up


def ik(p, s):
    """손끝을 p 에 두는 관절 각도 (범위 안의 해 중 어깨를 덜 꺾는 것).  없으면 None"""
    rel = np.asarray(p) - SHOULDER
    psi, u, v = np.arctan2(-rel[2], rel[0]), np.hypot(rel[0], rel[2]), rel[1]
    best = None
    for psi_, u_ in ((psi, u), (psi - np.sign(psi) * np.pi, -u)):        # 앞으로 뻗기 / 뒤로 넘기기
        c = np.clip((u_ * u_ + v * v - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
        for b in (np.arccos(c), -np.arccos(c)):
            a = np.arctan2(u_, v) - np.arctan2(L2 * np.sin(b), L1 + L2 * np.cos(b))
            q = np.array([psi_ * s[0], a * s[1], b * s[2]])
            if np.all(np.abs(q) <= LIM) and (best is None or abs(q[1]) < abs(best[1])):
                best = q
    return best


def calibrate(samples):
    """(관절 각도, 손끝 위치) 기록으로 관절 회전 방향 부호를 고른다"""
    best = min(itertools.product((1, -1), repeat=3),
               key=lambda s: np.mean([np.linalg.norm(fk(q, s) - t) for q, t in samples]))
    err = np.mean([np.linalg.norm(fk(q, best) - t) for q, t in samples])
    return np.array(best), err


# ---------------------------------------------------------------- 정책
class ScriptPolicy:
    """물체 위 -> 내려가기 -> 닫기 -> 들어올리기 -> 같은 색 상자 위 -> 열기"""
    def __init__(self, signs):
        self.s = signs

    def reset(self):
        self.phase = 0
        self.tgt = np.zeros(3)          # Unity 의 관절 목표 각도를 똑같이 따라 계산
        self.wait = 0

    def act(self, raw):
        tip, obj = np.array(raw["tip"]), np.array(raw["obj"])
        goal = goal_point(raw)
        grip = -1.0
        if self.phase == 0:                                   # 물체 위로
            wp = obj + [0, 0.5, 0]
            if np.linalg.norm(tip - wp) < 0.12:
                self.phase = 1
        if self.phase == 1:                                   # 내려가기
            wp = obj + [0, 0.15, 0]
            if np.linalg.norm(tip - obj) < 0.22:
                self.phase = 2
        if self.phase == 2:                                   # 닫기
            wp, grip = obj + [0, 0.15, 0], 1.0
            if raw["held"]:
                self.phase, self.lift = 3, np.array([tip[0], 1.1, tip[2]])
            else:
                self.wait += 1
                if self.wait > 3:                             # 못 잡았으면 다시 접근
                    self.phase, self.wait = 0, 0
        if self.phase == 3:                                   # 들어올리기
            wp, grip = self.lift, 1.0
            if np.linalg.norm(tip - wp) < 0.15:
                self.phase = 4
        if self.phase == 4:                                   # 물체가 상자 위에 오도록 (손끝-물체 어긋남 보정)
            wp, grip = goal - (obj - tip), 1.0
            if np.linalg.norm((obj - goal)[[0, 2]]) < 0.12:
                self.phase = 5
        if self.phase == 5:                                   # 놓기
            wp, grip = tip, -1.0
        q = ik(wp, self.s)
        if q is None:
            q = self.tgt
        d = np.clip((q - self.tgt) / MAX_DELTA, -1.0, 1.0)
        self.tgt = np.clip(self.tgt + d * MAX_DELTA, -LIM, LIM)
        return np.append(d, grip)


def run(env, name, episodes, policy=None):
    counts = {"place": 0, "wrong": 0, "lost": 0, "timeout": 0}
    totals, grabs = [], 0
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep)
        if policy:
            policy.reset()
        total = 0.0
        while True:
            a = policy.act(env.raw) if policy else env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        counts[info["outcome"]] += 1
        totals.append(total)
        grabs += info["grabbed"]
        print(f"[{name}] 에피소드 {ep:3d} | 보상 {total:7.2f} | 스텝 {env.steps:4d} | 잡아 봄 {info['grabbed']!s:5} "
              f"| 상자까지 {info['goal_dist']:.2f} | 결과: {info['outcome']}")
    print("-" * 70)
    print(f"[{name}] 성공 {counts['place']}  틀린 상자 {counts['wrong']}  잃어버림 {counts['lost']}  시간초과 {counts['timeout']}"
          f"  | 잡아 본 판 {grabs}  | 평균 보상 {np.mean(totals):.2f}")


def calibrate_signs(env):
    print("[보정] 관절 회전 방향을 확인합니다 (무작위로 조금 움직임)...")
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    samples = [(np.array(env.raw["joints"]), np.array(env.raw["tip"]))]
    for _ in range(15):
        env.step(np.append(rng.uniform(-1, 1, 3), -1.0))
        samples.append((np.array(env.raw["joints"]), np.array(env.raw["tip"])))
    signs, err = calibrate(samples)
    print(f"[보정] 부호 {signs.tolist()}  평균 오차 {err:.3f}" + ("" if err < 0.05 else "  <- 오차가 큼, 팔 길이/위치 설정 확인 필요"))
    return signs


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "script"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (10 if mode == "script" else 20)
    env = UnityPickEnv()
    try:
        if mode == "script":
            run(env, "script", n, ScriptPolicy(calibrate_signs(env)))
        elif mode == "random":
            run(env, "random", n)
        elif mode == "check":
            from stable_baselines3.common.env_checker import check_env
            check_env(env, warn=True)
            print("check_env 통과")
        else:
            print(__doc__)
    finally:
        env.close()
