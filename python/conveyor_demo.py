"""
conveyor_demo.py  -  컨베이어 연속 분류 시연

컨베이어 벨트로 물체(빨강/파랑)가 계속 들어오고, 팔이 하나씩 집어 같은 색 상자에 넣는다.
학습은 물체 1개짜리 pick 과제로 했고, 여기서는 그 정책을 물체마다 한 판씩 이어 붙여 실행한다.
한 물체를 8초 안에 못 잡으면 팔을 세우고 다시 시도한다 (최대 3번).  재시도 횟수는 마지막 집계에 따로 나온다.

  python conveyor_demo.py                  학습된 pick_sac 모델로 30개 처리
  python conveyor_demo.py 100              100개
  python conveyor_demo.py 30 ppo           pick_ppo 모델
  python conveyor_demo.py 30 v2            pick_sac_v2 모델 (팔 제자리 복귀 없이 바로 다음 물체로)
  python conveyor_demo.py 30 script        학습 없이 IK 규칙 정책으로 (비교용)
  Ctrl+C 로 언제든 멈추고 집계를 본다.

순서: 이 프로그램을 먼저 실행 -> Unity 에서 Play(▶).  눈으로 볼 때는 Ticks Per Frame=1.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import numpy as np

from pick_env import UnityPickEnv

STEP_SEC = 0.1          # 한 스텝의 시뮬레이션 시간
ATTEMPT_STEPS = 80      # 한 번 시도에 이만큼(8초) 넘게 걸리면 팔을 세우고 같은 물체를 다시 집는다 (정상은 20~35스텝)
MAX_ATTEMPTS = 3        # 마지막 시도는 300스텝(시간초과)까지 간다


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 30
    from pick_train import ALGOS, VARIANTS, run_name
    who = next((a for a in args if a in ALGOS or a == "script"), "sac")
    variant = next((a for a in args if a in VARIANTS), "full")

    env = UnityPickEnv(task="conveyor", **(VARIANTS[variant] if who != "script" else {}))
    try:
        if who == "script":
            from pick_main import ScriptPolicy, calibrate_signs
            policy = ScriptPolicy(calibrate_signs(env))
            act = lambda obs: policy.act(env.raw)
            reset_policy = policy.reset
        else:
            model = ALGOS[who].load(run_name(who, variant), device="cpu")
            who = run_name(who, variant)
            act = lambda obs: model.predict(obs, deterministic=True)[0]
            reset_policy = lambda: None

        counts = {"place": 0, "wrong": 0, "lost": 0, "timeout": 0}
        work_steps, retries, t0 = 0, 0, time.time()
        try:
            for k in range(1, n + 1):
                obs, info = env.reset()
                reset_policy()
                color = "빨강" if info["color"] == 0 else "파랑"
                start_obj = np.array(env.raw["obj"])
                steps, attempt = 0, 1
                while True:
                    obs, r, term, trunc, info = env.step(act(obs))
                    if term or trunc:
                        break
                    if env.steps >= ATTEMPT_STEPS and attempt < MAX_ATTEMPTS:
                        # 재시도: 팔을 세우고 같은 물체를 다시 집는다 (다른 자세에서 시작하므로 같은 실수를 되풀이하지 않는다)
                        steps += env.steps
                        obs, info = env.reset()
                        reset_policy()
                        if np.linalg.norm(np.array(env.raw["obj"]) - start_obj) > 0.2:   # 들다가 떨어뜨려 물체가 바뀜
                            print(f"[{who}] 물체 {k:3d} 재시도 대상이 바뀜: {np.round(start_obj, 2)} -> {np.round(env.raw['obj'], 2)}")
                            info = {"outcome": "lost"}     # 새 물체는 다음 차례에 다시 reset 해서 집는다
                            break
                        attempt += 1
                        retries += 1
                        print(f"[{who}] 물체 {k:3d} ({color}) | {ATTEMPT_STEPS}스텝 안에 못 잡음 -> 재시도 {attempt - 1}")
                steps += env.steps
                counts[info["outcome"]] += 1
                work_steps += steps
                extra = ""
                if info["outcome"] == "timeout":           # 왜 못 넣었는지 단서
                    o = env.raw["obj"]
                    extra = (f" | 잡아 봄 {info['grabbed']} 들고 있음 {bool(info['held'])} 상자까지 {info['goal_dist']:.2f}"
                             f" 물체 위치 ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f})")
                print(f"[{who}] 물체 {k:3d} ({color}) | {steps:3d}스텝 | 결과: {info['outcome']}{extra}")
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C")
        done = sum(counts.values())
        if done:
            per_min = counts["place"] / (work_steps * STEP_SEC / 60.0) if work_steps else 0.0
            print("-" * 70)
            print(f"[{who}] 처리 {done}개 | 바르게 분류 {counts['place']} | 잘못 분류 {counts['wrong']} | "
                  f"잃어버림 {counts['lost']} | 시간초과 {counts['timeout']} | 재시도 {retries}회")
            print(f"[{who}] 정확도 {counts['place'] / done:.0%} | 팔이 일한 시간 기준 분당 {per_min:.1f}개 "
                  f"(기다린 시간 제외) | 실제 경과 {time.time() - t0:.0f}초")
    finally:
        env.close()


if __name__ == "__main__":
    main()
