"""
arm_train.py  -  로봇 팔 에이전트를 학습시키고, 학습된 에이전트를 실행해 보는 프로그램

  python arm_train.py train                    PPO 로 학습 (기본 300000 스텝).  끝나면 arm_ppo.zip 저장
  python arm_train.py train 500000             스텝 수 지정
  python arm_train.py eval                     저장된 모델로 20판 실행

[알고리즘]  ppo(기본) / sac / td3 / a2c      (행동이 연속값이라 이 넷을 쓸 수 있다)
  python arm_train.py train 150000 sac         -> arm_sac.zip
  python arm_train.py eval sac

[실험용 변형]  상태/보상의 한 부분을 바꿔서 학습하고, 그 부분이 얼마나 중요한지 비교한다
  novel      관절 각속도를 숨김
  noshape    접근 보상(K_SHAPE) 없음  (성공 보상 +10 만 남음: 희소 보상)
  noenergy   행동 크기 감점(R_ENERGY) 없음
  python arm_train.py train 300000 noshape       -> arm_ppo_noshape.zip
  python arm_train.py eval noshape

학습 곡선은 logs/ 폴더에 저장되고, 평가 결과는 results.csv 에 한 줄씩 쌓인다 (plot_results.py 가 그래프/표로 만든다).
학습 중 Ctrl+C 로 멈춰도 그때까지의 모델을 저장한다.
순서: 이 프로그램을 먼저 실행 -> Unity 에서 Play(▶).  학습은 Ticks Per Frame=0, 눈으로 볼 때는 1.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import csv
import sys

from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

from arm_env import UnityArmEnv

VARIANTS = {
    "full": dict(),
    "novel": dict(use_vel=False),
    "noshape": dict(k_shape=0.0),
    "noenergy": dict(r_energy=0.0),
}
ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C}


def run_name(algo="ppo", variant="full"):
    return algo if variant == "full" else f"{algo}_{variant}"


def model_path(algo="ppo", variant="full"):
    return "arm_" + run_name(algo, variant)


def make_model(algo, env):
    if algo == "ppo":
        return PPO("MlpPolicy", env, n_steps=1024, batch_size=128, learning_rate=3e-4, gamma=0.99,
                   ent_coef=0.0, device="cpu", seed=0, verbose=1)
    if algo == "a2c":
        return A2C("MlpPolicy", env, n_steps=16, learning_rate=7e-4, gamma=0.99,
                   device="cpu", seed=0, verbose=1)
    if algo == "sac":
        return SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000, learning_starts=1_000,
                   batch_size=128, gamma=0.99, train_freq=1, gradient_steps=1, device="cpu", seed=0, verbose=1)
    if algo == "td3":
        return TD3("MlpPolicy", env, learning_rate=1e-3, buffer_size=100_000, learning_starts=1_000,
                   batch_size=128, gamma=0.99, train_freq=1, gradient_steps=1, device="cpu", seed=0, verbose=1)
    raise ValueError(algo)


def train(steps, variant="full", algo="ppo"):
    os.makedirs("logs", exist_ok=True)
    env = Monitor(UnityArmEnv(**VARIANTS[variant]), filename=os.path.join("logs", run_name(algo, variant)))
    try:
        model = make_model(algo, env)
        try:
            model.learn(total_timesteps=steps)
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C - 여기까지 학습한 모델을 저장합니다.")
        model.save(model_path(algo, variant))
        print(f"저장 완료: {model_path(algo, variant)}.zip  (학습 스텝 {model.num_timesteps})")
    finally:
        env.close()


def evaluate(variant="full", algo="ppo", episodes=20):
    env = UnityArmEnv(**VARIANTS[variant])
    model = ALGOS[algo].load(model_path(algo, variant), device="cpu")
    label = run_name(algo, variant)
    counts = {"reach": 0, "timeout": 0}
    totals, steps, dists, floors = [], [], [], []
    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=ep)                       # 다른 정책과 같은 seed -> 같은 목표
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
            dists.append(info["dist"])
            floors.append(info["floor_steps"])
            print(f"[{label}] 에피소드 {ep:3d} | 보상 합계 {total:7.2f} | 스텝 {env.steps:4d} | "
                  f"마지막 거리 {info['dist']:.2f} | 바닥 근처 {info['floor_steps']:3d}스텝 | 결과: {info['outcome']}")
        n = len(totals)
        mean = lambda xs: sum(xs) / len(xs)
        print("-" * 70)
        print(f"[{label}] 성공(reach) {counts['reach']}  시간초과(timeout) {counts['timeout']}"
              f"  | 평균 보상 {mean(totals):.2f}  | 평균 스텝 {mean(steps):.0f}  | 평균 마지막 거리 {mean(dists):.2f}  | 평균 바닥 근처 {mean(floors):.1f}스텝")
        save_result(label, counts, mean(totals), mean(steps), mean(dists), mean(floors), n)
    finally:
        env.close()


def save_result(label, counts, reward, steps, dist, floor, n):
    """평가 결과를 results.csv 에 한 줄 추가한다 (같은 이름이 있으면 새 값으로 바꾼다)."""
    path = "results.csv"
    header = ["name", "episodes", "reach", "timeout", "mean_reward", "mean_steps", "mean_final_dist", "mean_floor_steps"]
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)][1:]
    rows = [r for r in rows if r and r[0] != label]
    rows.append([label, n, counts["reach"], counts["timeout"], f"{reward:.2f}", f"{steps:.0f}", f"{dist:.2f}", f"{floor:.1f}"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[결과 저장] {path} 에 '{label}' 기록")


def parse(rest):
    steps = int(rest[0]) if rest and rest[0].isdigit() else 300_000
    variant = next((a for a in rest if a in VARIANTS), "full")
    algo = next((a for a in rest if a in ALGOS), "ppo")
    return steps, variant, algo


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    steps, variant, algo = parse(sys.argv[2:])
    if mode == "train":
        train(steps, variant, algo)
    elif mode == "eval":
        evaluate(variant, algo)
    else:
        print("사용법: python arm_train.py [train|eval] [스텝 수] [ppo|sac|td3|a2c] [full|novel|noshape|noenergy]")
