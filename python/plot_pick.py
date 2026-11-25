"""
plot_pick.py  -  집기(pick) 과제 학습 곡선 그래프 (학습이 끝난 뒤에 실행, Unity 필요 없음)

  python plot_pick.py

입력: logs/pick_sac.monitor.csv (v1),  logs/pick_sac_v2.monitor.csv (v2)
출력: curves_pick.png  (위: 에피소드 보상,  아래: 최근 100판 성공률·잡기 비율)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 100
RUNS = [("pick_sac", "v1 (기본)"), ("pick_sac_v2", "v2 (정밀·무작위 시작 자세)")]
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    df = pd.read_csv(os.path.join("logs", f"{name}.monitor.csv"), skiprows=1)
    df["steps"] = df["l"].cumsum()
    df["reward"] = df["r"].rolling(WINDOW, min_periods=1).mean()
    df["place"] = (df["outcome"] == "place").astype(float).rolling(WINDOW, min_periods=1).mean() * 100
    df["grab"] = df["grabbed"].astype(str).eq("True").astype(float).rolling(WINDOW, min_periods=1).mean() * 100
    return df


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for (name, label), color in zip(RUNS, ["tab:blue", "tab:orange"]):
        if not os.path.exists(os.path.join("logs", f"{name}.monitor.csv")):
            continue
        df = load(name)
        ax1.plot(df["steps"], df["reward"], color=color, label=label)
        ax2.plot(df["steps"], df["place"], color=color, label=f"{label} 성공률")
        ax2.plot(df["steps"], df["grab"], color=color, linestyle="--", alpha=0.6, label=f"{label} 잡기 비율")
    ax1.set_ylabel(f"에피소드 보상 (최근 {WINDOW}판 평균)")
    ax1.set_title("집어서 분류하기 학습 곡선 (SAC)")
    ax2.set_ylabel(f"비율 % (최근 {WINDOW}판)")
    ax2.set_xlabel("환경 스텝")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("curves_pick.png", dpi=150)
    print("[저장] curves_pick.png")


if __name__ == "__main__":
    main()
