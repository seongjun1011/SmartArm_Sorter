"""
pick_train.py  -  집기 에이전트를 학습시키고, 학습된 에이전트를 실행해 보는 프로그램

  python pick_train.py train                   SAC 로 학습 (기본 300000 스텝).  끝나면 pick_sac.zip 저장
  python pick_train.py train 500000 ppo        스텝 수 / 알고리즘 지정   -> pick_ppo.zip
  python pick_train.py train 200000 resume     저장된 pick_sac.zip 에 이어서 200000 스텝 더 (Ctrl+C 로 끊어도 이어 갈 수 있음)
  python pick_train.py eval                    저장된 모델로 20판 실행
  python pick_train.py eval ppo

[실험용 변형]
  noshape    단계별 거리 보상 없음 (성공 +10 / 실패 감점만)    -> pick_sac_noshape.zip
  v2         관절 12도/스텝(빠르게), 잡는 거리 0.22(정밀하게), 가운데 놓기 보너스 +5, 무작위 자세에서 시작
             python pick_train.py train 300000 v2   -> pick_sac_v2.zip

도달 과제(arm_*.zip, results.csv)와 파일 이름이 겹치지 않는다.  학습 곡선은 logs/pick_*.monitor.csv, 평가 결과는 pick_results.csv.
학습 중 Ctrl+C 로 멈춰도 그때까지의 모델을 저장한다.
순서: 이 프로그램을 먼저 실행 -> Unity 에서 Play(▶).  학습은 Ticks Per Frame=0, 눈으로 볼 때는 1.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import csv
import sys

from stable_baselines3.common.monitor import Monitor

from arm_train import ALGOS, make_model
from pick_env import UnityPickEnv

VARIANTS = {
    "full": dict(),
    "noshape": dict(k_shape=0.0),
    "v2": dict(max_delta=12.0, grab_dist=0.22, rand_pose=True, r_center=5.0),
}
OUTCOMES = ("place", "wrong", "lost", "timeout")


def run_name(algo="sac", variant="full"):
    return f"pick_{algo}" if variant == "full" else f"pick_{algo}_{variant}"


def train(steps, variant="full", algo="sac", resume=False):
    """resume=True 면 저장된 모델(과 SAC 경험 버퍼)을 불러와 이어서 steps 만큼 더 학습한다"""
    os.makedirs("logs", exist_ok=True)
    name = run_name(algo, variant)
    buffer = name + "_buffer.pkl"
    resume = resume and os.path.exists(name + ".zip")
    env = Monitor(UnityPickEnv(**VARIANTS[variant]), filename=os.path.join("logs", name),
                  info_keywords=("outcome", "grabbed"), override_existing=not resume)   # 이어서 학습하면 곡선도 이어 붙임
    try:
        if resume:
            model = ALGOS[algo].load(name, env=env, device="cpu")
            if hasattr(model, "load_replay_buffer") and os.path.exists(buffer):
                model.load_replay_buffer(buffer)
            print(f"[이어서 학습] {name}.zip  (지금까지 {model.num_timesteps}스텝)")
        else:
            model = make_model(algo, env)
        try:
            model.learn(total_timesteps=steps, reset_num_timesteps=not resume)
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C - 여기까지 학습한 모델을 저장합니다.")
        model.save(name)
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(buffer)
        print(f"저장 완료: {name}.zip  (학습 스텝 {model.num_timesteps})")
    finally:
        env.close()


def evaluate(variant="full", algo="sac", episodes=20):
    env = UnityPickEnv(**VARIANTS[variant])
    name = run_name(algo, variant)
    model = ALGOS[algo].load(name, device="cpu")
    counts = {k: 0 for k in OUTCOMES}
    totals, steps, centers, grabs = [], [], [], 0
    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=ep)                       # 같은 seed -> 같은 물체 위치/색
            total = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            counts[info["outcome"]] += 1
            totals.append(total)
            steps.append(env.steps)
            grabs += info["grabbed"]
            if info["outcome"] == "place":
                centers.append(info["center_err"])
            print(f"[{name}] 에피소드 {ep:3d} | 보상 합계 {total:7.2f} | 스텝 {env.steps:4d} | 잡아 봄 {info['grabbed']!s:5} "
                  f"| 상자까지 {info['goal_dist']:.2f} | 결과: {info['outcome']}")
        mean = lambda xs: sum(xs) / len(xs)
        print("-" * 70)
        print(f"[{name}] 성공 {counts['place']}  틀린 상자 {counts['wrong']}  잃어버림 {counts['lost']}  시간초과 {counts['timeout']}"
              f"  | 잡아 본 판 {grabs}  | 평균 보상 {mean(totals):.2f}  | 평균 스텝 {mean(steps):.0f}"
              f"  | 상자 가운데까지 평균 {mean(centers) if centers else float('nan'):.2f}")
        save_result(name, counts, grabs, mean(totals), mean(steps), len(totals), mean(centers) if centers else float("nan"))
    finally:
        env.close()


def save_result(name, counts, grabs, reward, steps, n, center):
    """평가 결과를 pick_results.csv 에 기록 (같은 이름이 있으면 새 값으로 바꾼다)"""
    path = "pick_results.csv"
    header = ["name", "episodes", *OUTCOMES, "grabbed", "mean_reward", "mean_steps", "mean_center_err"]
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
    rows = [r for r in rows if r and r[0] != name]
    rows.append([name, n, *[counts[k] for k in OUTCOMES], grabs, f"{reward:.2f}", f"{steps:.0f}", f"{center:.2f}"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[결과 저장] {path} 에 '{name}' 기록")


def parse(rest):
    steps = int(rest[0]) if rest and rest[0].isdigit() else 300_000
    variant = next((a for a in rest if a in VARIANTS), "full")
    algo = next((a for a in rest if a in ALGOS), "sac")
    return steps, variant, algo


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    steps, variant, algo = parse(sys.argv[2:])
    if mode == "train":
        train(steps, variant, algo, resume="resume" in sys.argv)
    elif mode == "eval":
        evaluate(variant, algo)
    else:
        print("사용법: python pick_train.py [train|eval] [스텝 수] [sac|ppo|td3|a2c] [full|noshape|v2]")
