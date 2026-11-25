"""
run_all.py  -  여러 실험을 차례로 학습 + 평가한다 (Unity 는 Play 상태로 계속 켜 둔다)

  python run_all.py                       기본 목록 전체, 각 300000 스텝
  python run_all.py 150000                각 150000 스텝
  python run_all.py 300000 ppo:novel sac 원하는 것만 (형식  알고리즘[:변형],  예: ppo  ppo:novel  ppo:noshape  sac  td3)

이미 학습된 모델(zip)이 있으면 학습은 건너뛰고 평가만 한다 -> 중간에 멈춰도 다시 실행하면 이어진다.
끝나면  python plot_results.py  로 그래프와 표를 만든다.
Unity 는 연결이 끊기면 자동으로 다시 접속을 시도하므로, Play 를 한 번만 누르면 된다. (Ticks Per Frame = 0 으로 두세요)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

from arm_train import ALGOS, VARIANTS, evaluate, model_path, train

DEFAULT = ["ppo", "ppo:novel", "ppo:noshape", "ppo:noenergy", "sac"]


def parse(item):
    algo, _, variant = item.partition(":")
    variant = variant or "full"
    if algo not in ALGOS or variant not in VARIANTS:
        raise SystemExit(f"알 수 없는 항목: {item}   (예: ppo  ppo:novel  ppo:noshape  sac  td3)")
    return algo, variant


if __name__ == "__main__":
    args = sys.argv[1:]
    steps = int(args[0]) if args and args[0].isdigit() else 300_000
    items = [a for a in args if not a.isdigit()] or DEFAULT
    plan = [parse(i) for i in items]
    print(f"[run_all] 실험 {len(plan)}개, 각 {steps} 스텝: " + ", ".join(f"{a}:{v}" for a, v in plan))
    t0 = time.time()
    for k, (algo, variant) in enumerate(plan, 1):
        print(f"\n===== [{k}/{len(plan)}] {algo}:{variant} =====")
        if os.path.exists(model_path(algo, variant) + ".zip"):
            print("이미 학습된 모델이 있어 학습을 건너뜁니다.")
        else:
            train(steps, variant, algo)
            time.sleep(2)                              # Unity 가 다시 접속할 시간
        evaluate(variant, algo)
        time.sleep(2)
        print(f"[run_all] 경과 {(time.time() - t0) / 60:.0f}분")
    print("\n[run_all] 전부 끝났습니다.  python plot_results.py 를 실행하세요.")
