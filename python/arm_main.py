"""
arm_main.py  -  환경 실행/검증용 프로그램

  python arm_main.py random          무작위 행동으로 20판            (바닥선: 아무것도 모르는 에이전트)
  python arm_main.py zero            아무 행동도 안 함 20판           (팔이 중력으로 처지는지, 가만히 있으면 점수가 얼마인지)
  python arm_main.py climb           관절을 하나씩 조금씩 움직이며 거리가 줄면 계속, 늘면 반대로 가는 규칙 20판
                                     (미분 없이 관측만 쓰는 손으로 만든 규칙: 이 문제가 풀 수 있는 문제인지 확인)
  python arm_main.py check           Gymnasium 규격 검사 (check_env)
  python arm_main.py random 50       50판

반드시 이 프로그램을 먼저 실행한 뒤, Unity 에서 Play(▶)를 누른다.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import numpy as np

from arm_env import UnityArmEnv


def random_policy(env, obs):
    return env.action_space.sample()


def zero_policy(env, obs):
    return np.zeros(3)


class ClimbPolicy:
    """관절 j 를 한 스텝씩 번갈아 움직인다.  직전 스텝에서 거리가 줄었으면 같은 방향, 늘었으면 방향을 바꾼다."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.j = 0
        self.d = np.ones(3)
        self.prev = None
        self.last = None

    def __call__(self, env, obs):
        dist = float(np.linalg.norm(obs[12:15])) * 2.6          # 정규화된 (목표-손끝) 벡터 -> 거리
        if self.prev is not None and self.last is not None:
            if dist > self.prev:
                self.d[self.last] *= -1                          # 나빠졌으면 그 관절의 방향을 바꿈
        self.prev = dist
        self.j = (self.j + 1) % 3
        self.last = self.j
        a = np.zeros(3)
        a[self.j] = self.d[self.j]
        return a


def run(make_policy, episodes, label):
    env = UnityArmEnv()
    counts = {"reach": 0, "timeout": 0}
    totals, steps, dists = [], [], []
    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=ep)
            policy = make_policy()
            total = 0.0
            while True:
                obs, reward, terminated, truncated, info = env.step(policy(env, obs))
                total += reward
                if terminated or truncated:
                    break
            counts[info["outcome"]] += 1
            totals.append(total)
            steps.append(env.steps)
            dists.append(info["dist"])
            print(f"[{label}] 에피소드 {ep:3d} | 보상 합계 {total:7.2f} | 스텝 {env.steps:4d} | "
                  f"마지막 거리 {info['dist']:.2f} | 바닥 근처 {info['floor_steps']:3d}스텝 | 결과: {info['outcome']}")
        n = len(totals)
        print("-" * 70)
        print(f"[{label}] 성공(reach) {counts['reach']}  시간초과(timeout) {counts['timeout']}"
              f"  | 평균 보상 {sum(totals) / n:.2f}  | 평균 스텝 {sum(steps) / n:.0f}  | 평균 마지막 거리 {sum(dists) / n:.2f}")
    finally:
        env.close()


def run_check():
    from gymnasium.utils.env_checker import check_env
    env = UnityArmEnv()
    try:
        check_env(env)
        print("check_env 통과: Gymnasium 규격에 맞는 환경입니다.")
    finally:
        env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if mode == "random":
        run(lambda: random_policy, n, "random")
    elif mode == "zero":
        run(lambda: zero_policy, n, "zero")
    elif mode == "climb":
        run(lambda: ClimbPolicy(), n, "climb")
    elif mode == "check":
        run_check()
    else:
        print("사용법: python arm_main.py [random|zero|climb|check] [에피소드 수]")
