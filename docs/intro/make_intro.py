"""Course introduction video (1920 x 1080, 30 fps, no sound) from the clips of record_clips.py.

  python3 docs/intro/make_intro.py              # everything -> outputs/intro/rl_shortcut_intro.mp4
  python3 docs/intro/make_intro.py 0 5          # only these scenes again, then join
  python3 docs/intro/make_intro.py --gifs       # the GIFs of the "0. Intro" Confluence page -> docs/lecture/l00_*.gif
  python3 docs/intro/make_intro.py --html       # clips and pictures of the HTML introduction -> docs/intro/html/media/
Same look as the PinkLAB Isaac Sim course video: dark background, PinkLAB logo, pinklab.art.
The logo is read from LOGO_PATH (not in this repository).
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "outputs/intro"
LECTURE = ROOT / "docs/lecture"
WORK = CLIPS / "work"
WORK.mkdir(parents=True, exist_ok=True)
LOGO_PATH = Path.home() / "doc_slide/pinklab_logo_text.png"
W, H, FPS = 1920, 1080, 30
BG, PANEL, PINK, WHITE, GRAY = (18, 19, 24), (30, 32, 40), (236, 112, 170), (245, 245, 247), (160, 164, 175)
BOLD, REGULAR = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf", "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
PX, PY, PW, PH = 70, 270, 1280, 720            # where the clip sits in a theme scene
SITE, REPO, MAIL = "pinklab.art", "github.com/PinkWink/rl_shortcut", "contact@pinklab.art"
THEME_SECONDS = 13.0


def font(size, bold=True):
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def video(name, start=0.0, length=None, label=None):
    """label: text shown on the clip (learning-progress clips). No ':' or ',' in it (ffmpeg filter syntax)."""
    return dict(kind="video", path=CLIPS / f"{name}.mp4", start=start, length=length, label=label)


def still(path):
    return dict(kind="still", path=LECTURE / path)


THEMES = [  # (title, pages, bullets, clips)
    ("환경 설정과 물리엔진", "1 ~ 3 페이지",
     ["폴더 하나에 설치: Isaac Sim · Isaac Lab", "USD 와 PhysX", "GPU 에서 로봇 4096 대를 동시에"],
     [still("raw/drop_1.png"), still("raw/parallel.png")]),
    ("역진자: 첫 강화학습", "4 · 5 페이지",
     ["PinkLAB 로터리 역진자 키트", "환경의 다섯 부분: 행동 · 관측 · 이벤트 · 보상 · 종료", "PPO 학습 약 20초", "보상과 관측을 바꿔 보는 실험"],
     [video("pendulum_kit", 0.0, 6.5), video("pendulum_many", 0.5, 6.5)]),
    ("역진자: 학습 과정", "5 페이지",
     ["키트 4096 개가 동시에 연습", "학습 전에는 곧바로 쓰러짐", "40 iteration 쯤 세우는 법을 찾음", "150 iteration 에 약 20초"],
     [video("pendulum_it0", 0.3, 4.3, "학습 전 · iteration 0"), video("pendulum_it20", 0.3, 4.3, "iteration 20"),
      video("pendulum_it149", 0.3, 4.4, "학습 후 · iteration 149")]),
    ("Pinky Pro: YOLO 도로 인식", "6 페이지",
     ["작은 도시 핑크시티", "카메라 영상과 정답을 자동으로 모으기", "YOLO 세그먼테이션 학습 약 70초", "도로 마스크를 8 × 16 칸의 도로 시야로"],
     [video("pinky_city", 2.0, 13.0), still("raw/pinky_check_3rows.png")]),
    ("Pinky Pro: 강화학습으로 길 찾기", "7 페이지",
     ["도로 시야만 보고 바퀴 속도를 결정", "학습 전에는 곧 도로를 벗어남", "50 iteration 이면 도로 안에서 달림", "300 iteration 2.9분이면 오른쪽 차로 유지"],
     [video("pinky_it0", 0.5, 8.0, "학습 전 · iteration 0"), video("pinky_it50", 0.5, 8.0, "iteration 50"),
      video("pinky_it299", 0.5, 10.0, "학습 후 · iteration 299")]),
    ("Pinky Pro: 카메라 + YOLO 로 주행", "7 페이지",
     ["학습은 카메라 없이 1024 대로", "평가는 로봇 카메라 + YOLO 로", "정책에게는 같은 모양의 관측", "16대 30초: 도로 이탈 없음"],
     [video("pinky_yolo", 4.0, 26.0), video("pinky_train", 1.0, 13.0)]),
    ("Unitree Go2: 4족 보행", "8 ~ 10 페이지",
     ["USD 에서 서 있는 로봇까지", "속도 명령을 따라 걷는 환경", "4096 대로 2.7분 학습", "명령한 속도와 실제 속도 비교"],
     [video("go2_forward", 1.0, 6.5), video("go2_play", 1.0, 13.0)]),
    ("Unitree G1: 모션 모방", "11 · 12 페이지",
     ["사람의 동작을 본보기로", "앞으로 걷기", "걷기 정책에게 쿵푸를 시키면?", "이소룡 쿵푸 학습과 추론"],
     [video("g1_walk", 1.0, 4.3), video("g1_walk_on_bruce_lee", 0.0, 4.3), video("g1_bruce_lee", 0.5, 8.6)])]
HERO = [video("pendulum_kit", 0.0, 10.0), video("pinky_city", 2.0, 20.0), video("go2_play", 1.0, 10.0),
        video("g1_walk", 1.0, 8.0), video("pinky_train", 1.0, 13.0), video("g1_bruce_lee", 0.5, 10.0)]
PAGES = ["소개 · 커리큘럼", "환경 설정", "Isaac Sim 물리엔진", "역진자 (1) 환경 만들기", "역진자 (2) PPO 학습과 평가",
         "Pinky Pro (1) YOLO 도로 인식", "Pinky Pro (2) 순환 도로 주행", "Go2 (1) USD 에서 서기까지", "Go2 (2) 걷기 환경",
         "Go2 (3) 학습과 평가", "G1 (1) 모션 모방으로 걷기", "G1 (2) 이소룡 쿵푸"]

LOGO = Image.open(LOGO_PATH).convert("RGBA")       # black letters: put it on a white plate


def paste_logo(image, center_x, top, width):
    logo = LOGO.resize((width, int(LOGO.height * width / LOGO.width)), Image.LANCZOS)
    pad_x, pad_y = int(width * 0.09), int(logo.height * 0.30)
    box = (int(center_x - width / 2 - pad_x), top, int(center_x + width / 2 + pad_x), top + logo.height + 2 * pad_y)
    fill = (255, 255, 255, 255) if image.mode == "RGBA" else (255, 255, 255)
    ImageDraw.Draw(image).rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill)
    image.paste(logo, (int(center_x - width / 2), top + pad_y), logo)
    return box[3]


def text_center(draw, y, text, size, color, bold=True):
    draw.text(((W - draw.textlength(text, font=font(size, bold))) / 2, y), text, font=font(size, bold), fill=color)


def card(lines, name, logo=(90, 420), site=(980, 44)):
    """A text-only scene. lines: [(y, text, size, colour, bold)]."""
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    paste_logo(image, W / 2, logo[0], logo[1])
    if site:
        text_center(draw, site[0], SITE, site[1], PINK)
    for y, text, size, color, bold in lines:
        text_center(draw, y, text, size, color, bold)
    image.save(WORK / name)
    return WORK / name


def curriculum_card(name):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((110, 70), "커리큘럼", font=font(84), fill=WHITE)
    draw.text((470, 100), "12 페이지 · 2일", font=font(46, False), fill=PINK)
    bottom = paste_logo(image, 1640, 60, 330)
    draw.text((1640 - draw.textlength(SITE, font=font(40)) / 2, bottom + 14), SITE, font=font(40), fill=PINK)
    colors = [GRAY, GRAY, (120, 160, 235), (240, 200, 110), (240, 200, 110), (190, 150, 235), (190, 150, 235),
              (110, 205, 190), (110, 205, 190), (110, 205, 190), (240, 130, 130), (240, 130, 130)]
    for k, title in enumerate(PAGES):
        x, y = 110 + (k // 6) * 880, 290 + (k % 6) * 118
        draw.rounded_rectangle((x, y, x + 820, y + 92), radius=18, fill=PANEL)
        draw.rounded_rectangle((x, y, x + 110, y + 92), radius=18, fill=colors[k])
        number = f"{k + 1:02d}"
        draw.text((x + 55 - draw.textlength(number, font=font(46)) / 2, y + 22), number, font=font(46), fill=BG)
        draw.text((x + 140, y + 24), title, font=font(42, False), fill=WHITE)
    image.save(WORK / name)
    return WORK / name


def schedule_card(name):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((110, 70), "2일 일정", font=font(84), fill=WHITE)
    draw.text((110 + draw.textlength("2일 일정", font=font(84)) + 30, 100), "예시", font=font(46, False), fill=PINK)
    bottom = paste_logo(image, 1640, 60, 330)
    draw.text((1640 - draw.textlength(SITE, font=font(40)) / 2, bottom + 14), SITE, font=font(40), fill=PINK)
    days = [("1일차", "", [("설치 · 물리엔진", GRAY, 120), ("역진자", (240, 200, 110), 70),
                                     ("Pinky Pro 도로 인식 · 주행", (190, 150, 235), 140), ("Go2 USD", (110, 205, 190), 50)]),
            ("2일차", "", [("Go2 걷기 환경 · 학습", (110, 205, 190), 90), ("G1 걷기 · 쿵푸", (240, 130, 130), 100)])]
    for d, (day, hours, blocks) in enumerate(days):
        y = 330 + d * 300
        draw.text((110, y), day, font=font(60), fill=WHITE)
        draw.text((110 + draw.textlength(day, font=font(60)) + 30, y + 14), hours, font=font(40, False), fill=GRAY)
        x, total = 110, sum(m for _, _, m in blocks)
        for label, color, minutes in blocks:
            w = minutes / total * 1700
            draw.rounded_rectangle((x, y + 100, x + w - 12, y + 210), radius=18, fill=color)
            size = 38 if draw.textlength(label, font=font(38)) < w - 40 else 30
            draw.text((x + (w - 12 - draw.textlength(label, font=font(size))) / 2, y + 155 - size / 2), label, font=font(size), fill=BG)
            x += w
    draw.text((110, 950), "오래 걸리는 학습은 걸어 두고, 미리 학습한 정책(checkpoints/)으로 바로 평가", font=font(40, False), fill=WHITE)
    image.save(WORK / name)
    return WORK / name


def theme_overlay(index, title, pages, bullets, name):
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((70, 66), f"{index:02d}", font=font(110), fill=PINK)
    draw.text((230, 70), title, font=font(72), fill=WHITE)
    width = draw.textlength(pages, font=font(34))
    draw.rounded_rectangle((232, 170, 232 + width + 44, 226), radius=28, outline=PINK, width=3)
    draw.text((254, 180), pages, font=font(34), fill=PINK)
    draw.rectangle((PX - 4, PY - 4, PX + PW + 3, PY + PH + 3), outline=(60, 63, 75), width=4)
    y = 330
    for bullet in bullets:
        draw.ellipse((1410, y + 16, 1426, y + 32), fill=PINK)
        line, rows = "", []
        for word in bullet.split(" "):                   # wrap to the width of the right column
            if draw.textlength((line + " " + word).strip(), font=font(38, False)) > 410 and line:
                rows.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        rows.append(line)
        for row in rows:
            draw.text((1446, y), row, font=font(38, False), fill=WHITE)
            y += 52
        y += 34
    draw.text((1410, 1000), f"{index} / {len(THEMES)}", font=font(28, False), fill=GRAY)
    draw.text((70, 1014), "핑크랩 강화학습 단기 과정  ·  NVIDIA Isaac Sim + Isaac Lab", font=font(28, False), fill=GRAY)
    bottom = paste_logo(image, 1640, 52, 330)
    draw.text((1640 - draw.textlength(SITE, font=font(40)) / 2, bottom + 14), SITE, font=font(40), fill=PINK)
    image.save(WORK / name)
    return WORK / name


def source(clip, seconds, size, fill=False):
    """ffmpeg input options and the filter that fits this input into `size` for `seconds`."""
    w, h = size
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}" if fill
           else f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x1e2028")
    if clip["kind"] == "video":
        speed = clip["length"] / seconds
        text = (f",drawtext=fontfile={BOLD}:text='{clip['label']}':x=28:y=26:fontsize=46:fontcolor=white:box=1"
                ":boxcolor=0x121318@0.85:boxborderw=18") if clip.get("label") else ""
        return (["-ss", str(clip["start"]), "-t", str(clip["length"]), "-i", str(clip["path"])],
                f"setpts=PTS/{speed:.4f},fps={FPS},{fit},setsar=1{text}")
    frames = int(seconds * FPS)
    zoom = f"zoompan=z='1+0.06*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={FPS}"
    pre = f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w * 2}:{h * 2}:(ow-iw)/2:(oh-ih)/2:color=0x1e2028"
    return ["-i", str(clip["path"])], f"{pre},{zoom},setsar=1"


def encode(inputs, graph, seconds, out):
    fade = f"fade=t=in:st=0:d=0.4,fade=t=out:st={seconds - 0.4:.2f}:d=0.4,format=yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex", graph + f",{fade}[out]", "-map", "[out]",
                    "-t", f"{seconds:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def scene_card(image, seconds, out):
    encode(["-loop", "1", "-t", str(seconds), "-i", str(image)], f"[0:v]fps={FPS},scale={W}:{H}", seconds, out)


def scene_theme(index, out):
    title, pages, bullets, clips = THEMES[index - 1]
    overlay = theme_overlay(index, title, pages, bullets, f"theme_{index:02d}.png")
    total = sum(c.get("length") or 1.0 for c in clips)
    inputs, parts = ["-f", "lavfi", "-t", str(THEME_SECONDS), "-i", f"color=c=0x121318:s={W}x{H}:r={FPS}"], []
    for number, clip in enumerate(clips, 1):
        share = (clip.get("length") or 1.0) / total if all(c["kind"] == "video" for c in clips) else 1 / len(clips)
        options, chain = source(clip, THEME_SECONDS * share, (PW, PH))
        inputs += options
        parts.append(f"[{number}:v]{chain}[c{number}]")
    inputs += ["-loop", "1", "-t", str(THEME_SECONDS), "-i", str(overlay)]
    joined = "".join(f"[c{n}]" for n in range(1, len(clips) + 1)) + f"concat=n={len(clips)}:v=1:a=0[clip]"
    graph = ";".join(parts) + f";{joined};[0:v][clip]overlay={PX}:{PY}:shortest=0[a];[a][{len(clips) + 1}:v]overlay=0:0"
    encode(inputs, graph, THEME_SECONDS, out)


def scene_hero(seconds, out):
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 720, W, H), fill=BG + (255,))
    text_center(draw, 760, "보상만 정해 주면, 로봇이 스스로 익히는 강화학습", 62, WHITE)
    text_center(draw, 856, "로터리 역진자  ·  Pinky Pro  ·  Unitree Go2  ·  Unitree G1", 44, PINK, False)
    paste_logo(image, W / 2 - 190, 946, 300)
    draw.text((W / 2 + 40, 968), SITE, font=font(56), fill=PINK)
    image.save(WORK / "hero.png")
    inputs, parts = ["-f", "lavfi", "-t", str(seconds), "-i", f"color=c=0x121318:s={W}x{H}:r={FPS}"], []
    for number, clip in enumerate(HERO, 1):
        options, chain = source(clip, seconds, (640, 360), fill=True)
        inputs += options
        parts.append(f"[{number}:v]{chain}[c{number}]")
    inputs += ["-loop", "1", "-t", str(seconds), "-i", str(WORK / "hero.png")]
    layout = "|".join(f"{(i % 3) * 640}_{(i // 3) * 360}" for i in range(6))
    graph = (";".join(parts) + ";" + "".join(f"[c{n}]" for n in range(1, 7))
             + f"xstack=inputs=6:layout={layout}[grid];[0:v][grid]overlay=0:0[a];[a][7:v]overlay=0:0")
    encode(inputs, graph, seconds, out)


SCENES = [  # (name, seconds, maker)
    ("title", 7.0, lambda out: scene_card(card(
        [(400, "강화학습 단기 과정", 130, WHITE, True), (600, "NVIDIA Isaac Sim · Isaac Lab 으로 2일", 64, WHITE, False),
         (710, "역진자  ·  Pinky Pro  ·  Unitree Go2  ·  Unitree G1", 56, WHITE, True)],
        "title.png", logo=(110, 640), site=(900, 72)), 7.0, out)),
    ("hero", 10.0, lambda out: scene_hero(10.0, out)),
    ("about", 10.0, lambda out: scene_card(card(
        [(300, "같은 틀을 점점 어려운 로봇에", 84, WHITE, True), (460, "환경의 다섯 부분 + PPO 를 네 가지 로봇에 적용", 54, WHITE, False),
         (545, "설치부터 학습 · 평가까지, 터미널 명령 그대로", 54, WHITE, False), (690, "학습 시간은 20초에서 40분까지 (RTX 5060 Ti 실측)", 54, WHITE, True),
         (775, "기다리지 않도록 미리 학습한 정책 포함", 46, GRAY, False)], "about.png", site=(940, 64)), 10.0, out)),
    ("curriculum", 11.0, lambda out: scene_card(curriculum_card("curriculum.png"), 11.0, out)),
]
for number in range(1, len(THEMES) + 1):
    SCENES.append((f"theme{number:02d}", THEME_SECONDS, lambda out, number=number: scene_theme(number, out)))
SCENES += [
    ("schedule", 10.0, lambda out: scene_card(schedule_card("schedule.png"), 10.0, out)),
    ("setup", 9.0, lambda out: scene_card(card(
        [(330, "준비물", 84, WHITE, True), (480, "Ubuntu 24.04  ·  NVIDIA RTX GPU (VRAM 8 GB 이상 권장)", 54, WHITE, False),
         (565, "NVIDIA Isaac Sim  ·  Isaac Lab", 54, WHITE, False),
         (700, "모든 것을 폴더 하나(~/rl_shortcut)에 설치", 54, WHITE, True)], "setup.png", site=(940, 64)), 9.0, out)),
    ("code", 9.0, lambda out: scene_card(card(
        [(360, "코드와 학습된 정책 공개", 96, WHITE, True), (540, REPO, 84, PINK, True),
         (700, "강의 자료의 명령을 그대로 따라 할 수 있음", 50, GRAY, False)], "code.png", site=(900, 72)), 9.0, out)),
    ("contact", 11.0, lambda out: scene_card(card(
        [(360, "이 과정을 수업하고 싶은 기관은", 72, WHITE, True), (460, "연락 주세요", 72, WHITE, True),
         (610, MAIL, 116, PINK, True), (800, "홈페이지", 40, GRAY, False), (858, SITE, 104, WHITE, True)],
        "contact.png", logo=(80, 620), site=None), 11.0, out)),
]


def gifs():
    """Short GIFs for the Confluence intro page (720 px wide, 8 fps)."""
    jobs = {"l00_hero.gif": None,
            "l00_pendulum.gif": [video("pendulum_kit", 0.0, 4.0), video("pendulum_many", 0.5, 4.0)],
            "l00_pinky.gif": [video("pinky_city", 2.0, 18.0), video("pinky_train", 1.0, 9.0)],
            "l00_go2.gif": [video("go2_forward", 1.0, 3.5), video("go2_play", 1.0, 5.0)],
            "l00_g1.gif": [video("g1_walk", 1.0, 3.5), video("g1_bruce_lee", 1.5, 7.0)]}
    for name, clips in jobs.items():
        out = LECTURE / name
        if clips is None:                                  # the hero scene of the video
            cmd = ["-i", str(WORK / "01_hero.mp4"), "-filter_complex",
                   "[0:v]fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"]
        else:
            inputs, parts = [], []
            for n, clip in enumerate(clips):
                seconds = clip["length"] / (2.0 if "pinky" in str(clip["path"]) else 1.0)
                options, chain = source(clip, seconds, (720, 405), fill=True)
                inputs += options
                parts.append(f"[{n}:v]{chain.replace(f'fps={FPS}', 'fps=8')}[c{n}]")
            graph = (";".join(parts) + ";" + "".join(f"[c{n}]" for n in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,"
                     "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none")
            cmd = inputs + ["-filter_complex", graph]
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + cmd + [str(out)], check=True)
        print(out.name, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


HTML_CLIPS = {  # name: (clip, playback speed)  -> docs/intro/html/media/<name>.mp4, 960 x 540, looped by the page
    "pendulum_kit": (video("pendulum_kit", 0.0, 7.0), 1.0), "pendulum_many": (video("pendulum_many", 0.5, 7.0), 1.0),
    "pendulum_it0": (video("pendulum_it0", 0.3, 5.0), 1.0), "pendulum_it20": (video("pendulum_it20", 0.3, 5.0), 1.0),
    "pendulum_it149": (video("pendulum_it149", 0.3, 5.0), 1.0),
    "pinky_city": (video("pinky_city", 2.0, 18.0), 2.0), "pinky_yolo": (video("pinky_yolo", 4.0, 28.0), 2.0),
    "pinky_train": (video("pinky_train", 1.0, 12.0), 1.5),
    "pinky_it0": (video("pinky_it0", 0.5, 8.0), 1.0), "pinky_it50": (video("pinky_it50", 0.5, 8.0), 1.0),
    "pinky_it299": (video("pinky_it299", 0.5, 10.0), 1.0),
    "go2_forward": (video("go2_forward", 1.0, 7.0), 1.0), "go2_play": (video("go2_play", 1.0, 10.0), 1.0),
    "g1_walk": (video("g1_walk", 1.0, 7.0), 1.0), "g1_walk_on_bruce_lee": (video("g1_walk_on_bruce_lee", 0.0, 4.5), 1.0),
    "g1_bruce_lee": (video("g1_bruce_lee", 0.5, 10.0), 1.0)}


def html_media():
    """Clips and pictures of the HTML introduction (docs/intro/html/index.html)."""
    media = ROOT / "docs/intro/html/media"
    media.mkdir(parents=True, exist_ok=True)
    x264 = ["-an", "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    for name, (clip, speed) in HTML_CLIPS.items():
        size = (1440, 540) if name == "pinky_yolo" else (960, 540)        # the YOLO panel is 8:3
        options, chain = source(clip, clip["length"] / speed, size, fill=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + options + ["-vf", chain] + x264 + [str(media / f"{name}.mp4")], check=True)
    # the six-clip grid of the hero scene, without the text band below it
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(WORK / "01_hero.mp4"), "-vf", "crop=1920:720:0:0,scale=1440:540"]
                   + x264 + [str(media / "hero.mp4")], check=True)
    for name, path, width in [("drop.jpg", LECTURE / "raw/drop_1.png", 960), ("parallel.jpg", LECTURE / "raw/parallel.png", 960),
                              ("yolo_check.jpg", LECTURE / "raw/pinky_check_3rows.png", 1280)]:
        image = Image.open(path).convert("RGB")
        image.resize((width, image.height * width // image.width), Image.LANCZOS).save(media / name, quality=86)
    logo = LOGO.resize((600, LOGO.height * 600 // LOGO.width), Image.LANCZOS)
    logo.save(media / "pinklab_logo.png")
    # index.html = page.html (the published page body) inside a document, for opening the folder locally
    html = media.parent
    reset = "<style>body{margin:0}img{max-width:100%}</style>"
    (html / "index.html").write_text('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" '
                                     f'content="width=device-width, initial-scale=1">\n{reset}\n</head>\n<body>\n'
                                     + (html / "page.html").read_text() + "</body>\n</html>\n")
    for f in sorted(media.iterdir()):
        print(f.name, f"{f.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    if "--gifs" in sys.argv:
        gifs()
        sys.exit()
    if "--html" in sys.argv:
        html_media()
        sys.exit()
    picked = {int(a) for a in sys.argv[1:]}
    for index, (name, seconds, make) in enumerate(SCENES):
        out = WORK / f"{index:02d}_{name}.mp4"
        if not picked or index in picked or not out.exists():
            make(out)
            print(f"[{index:02d}] {name}  {seconds:g} s", flush=True)
    listing = WORK / "list.txt"
    listing.write_text("".join(f"file '{WORK / f'{i:02d}_{name}.mp4'}'\n" for i, (name, _, _) in enumerate(SCENES)))
    final = CLIPS / "rl_shortcut_intro.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True)
    print(final, f"{sum(s for _, s, _ in SCENES):g} s, {final.stat().st_size / 1e6:.0f} MB")
