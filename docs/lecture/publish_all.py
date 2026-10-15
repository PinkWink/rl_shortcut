"""Create or update all lecture pages under the course's Confluence parent page.

  set -a; source <token env file>; set +a
  python3 docs/lecture/publish_all.py            # all pages
  python3 docs/lecture/publish_all.py 1 10       # only pages 1 and 10
Page ids are kept in pages.json (written when a page is first created).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT, SPACE = "3727785992", "2517565443"
PAGES = [
    ("lesson01_intro.md", "강화학습 단기 과정: 소개와 커리큘럼"),
    ("lesson02_setup.md", "환경 설정 (Ubuntu 24.04 / venv / Isaac Sim 5.1.0 / Isaac Lab 2.3.2)"),
    ("lesson03_physics.md", "Isaac Sim 물리엔진: USD · PhysX · GPU 병렬 시뮬레이션"),
    ("lesson04_cartpole_env.md", "역진자 (1): 강화학습 환경 만들기"),
    ("lesson05_cartpole_train.md", "역진자 (2): PPO 학습과 평가"),
    ("lesson06_go2_usd.md", "Unitree Go2 (1): USD 에서 서 있는 로봇까지"),
    ("lesson07_go2_env.md", "Unitree Go2 (2): 걷기 환경 - 명령 · 관측 · 보상"),
    ("lesson08_go2_train.md", "Unitree Go2 (3): 학습과 평가"),
    ("lesson09_g1_walk.md", "Unitree G1 (1): 모션 모방으로 앞으로 걷기"),
    ("lesson10_g1_kungfu.md", "Unitree G1 (2): 걷기 모델에게 쿵푸를? 이소룡 쿵푸 학습과 추론"),
]


def main():
    ids_file = HERE / "pages.json"
    ids = json.loads(ids_file.read_text()) if ids_file.exists() else {}
    only = {int(a) for a in sys.argv[1:]}
    for n, (md, title) in enumerate(PAGES, 1):
        if only and n not in only:
            continue
        pngs = sorted(set(re.findall(r"\((l\d+_[^)]+\.png)\)", (HERE / md).read_text())))
        cmd = ["python3", str(HERE / "publish_confluence.py"), str(HERE / md), title]
        cmd += ["--update", ids[md]] if md in ids else [PARENT, SPACE]
        out = subprocess.run(cmd + [str(HERE / p) for p in pngs], capture_output=True, text=True, check=True).stdout
        print(out.splitlines()[0])
        m = re.search(r"created page (\d+)", out)
        if m:
            ids[md] = m.group(1)
            ids_file.write_text(json.dumps(ids, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
