"""
plot_results.py  -  학습 곡선 그래프와 비교 표를 만든다 (학습/평가가 끝난 뒤에 실행, Unity 필요 없음)

  python plot_results.py

입력: logs/*.monitor.csv (학습 곡선),  results.csv (평가 결과)
출력: curves_variants.png  (PPO 변형 실험 곡선),  curves_algos.png  (알고리즘 비교 곡선),  results_table.md  (표)
필요: pip install matplotlib pandas
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 50      # 최근 50판 이동평균
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]   # 한글 글꼴 (Windows 는 맑은 고딕)
plt.rcParams["axes.unicode_minus"] = False


def load_curve(path):
    df = pd.read_csv(path, skiprows=1)                      # 첫 줄은 설명(#{...})
    df["steps"] = df["l"].cumsum()                          # 지금까지 쌓인 환경 스텝 수
    df["smooth"] = df["r"].rolling(WINDOW, min_periods=1).mean()
    return df


def curves(names, title, out):
    plt.figure(figsize=(8, 4.5))
    drawn = 0
    for name in names:
        p = os.path.join("logs", f"{name}.monitor.csv")
        if not os.path.exists(p):
            continue
        df = load_curve(p)
        plt.plot(df["steps"], df["smooth"], label=name)
        drawn += 1
    if drawn == 0:
        plt.close()
        print(f"[건너뜀] {out}: 해당하는 로그가 없습니다.")
        return
    plt.xlabel("환경 스텝")
    plt.ylabel(f"에피소드 보상 (최근 {WINDOW}판 평균)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[저장] {out}")


def table(out):
    if not os.path.exists("results.csv"):
        print("[건너뜀] results.csv 가 없습니다. 먼저 eval 을 실행하세요.")
        return
    with open("results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    names = ["이름", "판 수", "성공", "시간초과", "평균 보상", "평균 스텝", "평균 마지막 거리", "평균 바닥 근처 스텝"]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[저장] {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    # 변형 실험 곡선은 PPO 로 학습한 것만, 알고리즘 비교는 기본 설계(full)만
    curves(["ppo", "ppo_novel", "ppo_noshape", "ppo_noenergy"], "PPO 변형 실험 학습 곡선", "curves_variants.png")
    curves(["ppo", "sac", "td3", "a2c"], "알고리즘 비교 학습 곡선", "curves_algos.png")
    table("results_table.md")
