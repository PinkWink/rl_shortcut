"""Check the lecture pages against the repository (Isaac Sim is not needed).

  python3 docs/lecture/check_lessons.py              # code excerpts: every line must be in the file it names
  python3 docs/lecture/check_lessons.py --commands   # print every terminal command of every page, in course order
Code blocks start with a path comment ("# rl_shortcut/pendulum_env.py  (발췌)"); lines of "..." and trailing
comments are skipped, and a page line may be a part of a code line.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORDER = ["02_setup", "03_physics", "04_pendulum_env", "05_pendulum_train", "11_pinky_yolo", "12_pinky_drive",
         "06_go2_usd", "07_go2_env", "08_go2_train", "09_g1_walk", "10_g1_kungfu"]        # the order of the course


def blocks(page):
    """(heading, language, lines) of every fenced block."""
    heading, lang, body = "", None, []
    for line in (HERE / f"lesson{page}.md").read_text().splitlines():
        if lang is None:
            if line.startswith("#"):
                heading = line.lstrip("# ")
            elif line.startswith("```"):
                lang, body = line[3:].strip() or "text", []
        elif line.startswith("```"):
            yield heading, lang, body
            lang = None
        else:
            body.append(line)


def squeeze(line):
    """Without the trailing comment and extra spaces: the pages add or shorten comments."""
    return " ".join(re.sub(r"\s+#( .*)?$", "", line).split())


def commands():
    for page in ORDER:
        print(f"\n######## lesson{page}.md")
        for heading, lang, body in blocks(page):
            if lang == "bash":
                print(f"--- {heading}")
                print("\n".join(body))


def excerpts():
    bad = 0
    for page in ORDER:
        for heading, lang, body in blocks(page):
            if lang == "bash":       # a new terminal must work: the work folder first, then the course environment
                if body[:1] != ["cd ~/rl_shortcut"] and body[:1] != ["cd ~"]:
                    print(f"lesson{page}.md [{heading}]: bash block does not start with cd ~/rl_shortcut")
                    bad += 1
                if any(line.startswith("python ") for line in body) and body[1:2] != ["source scripts/env.sh"]:
                    print(f"lesson{page}.md [{heading}]: python command without source scripts/env.sh")
                    bad += 1
                continue
            if not body:
                continue
            m = re.match(r"\s*(?:#|//)\s*([\w./-]+\.\w+)", body[0])
            if not m:
                print(f"lesson{page}.md [{heading}]: {lang} block without a path comment: {body[0][:70]}")
                continue
            path = ROOT / m.group(1)
            if not path.exists():
                print(f"lesson{page}.md [{heading}]: no such file {m.group(1)}")
                bad += 1
                continue
            source = [squeeze(s) for s in path.read_text().splitlines()]
            for line in body[1:]:
                text = squeeze(line).removesuffix(" ...")          # a shortened line
                if text and not set(text) <= set(". ") and not any(text in s for s in source):
                    print(f"lesson{page}.md [{heading}] {m.group(1)}: not in the file: {line.strip()[:110]}")
                    bad += 1
    print(f"{bad} excerpt lines differ from the code")
    return bad


if __name__ == "__main__":
    commands() if "--commands" in sys.argv else sys.exit(1 if excerpts() else 0)
