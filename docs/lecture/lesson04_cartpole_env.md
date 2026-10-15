## 1. 역진자 (Cartpole)

### 과제

![카트를 밀어 막대를 세운다](l04_task.png){width=1000}

- 레일 위의 카트를 좌우로 밀어서, 카트에 핀으로 달린 막대(pole)가 쓰러지지 않게 합니다.
- 막대에는 모터가 없습니다. 정책이 할 수 있는 일은 **카트에 힘을 주는 것 하나**뿐입니다.

### 강화학습의 한 바퀴

![관측 → 행동 → 보상 → 다음 관측](l04_loop.png){width=1000}

- 모방학습(앞 과정)은 사람의 시연을 정답으로 삼았습니다. 강화학습에는 정답 행동이 없고, **보상(점수)** 만 있습니다.
- 정책은 수많은 시도 중 점수가 높았던 행동을 더 자주 하도록 바뀝니다. 그래서 "무엇이 좋은가"를 보상으로 잘 적는 것이 환경 설계의 핵심입니다.

### 환경을 이루는 다섯 부분

![Isaac Lab ManagerBasedRLEnv 의 구성](l04_managers.png){width=1000}

- Isaac Lab 의 `ManagerBasedRLEnv` 는 장면(scene) 위에 다섯 가지 설정을 얹어 환경을 만듭니다. 각 부분을 맡은 매니저(Manager)가 있어 이름이 그렇습니다.
- 오늘 만드는 역진자 · Go2 · G1 환경이 모두 같은 틀입니다. 바뀌는 것은 다섯 칸에 넣는 내용뿐입니다.

## 2. 코드: rl_shortcut/cartpole_env.py

### 장면 (Scene)

```python
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
```

- `CARTPOLE_CFG` 는 Isaac Lab 이 제공하는 역진자 USD(`cartpole.usd`)와 모터 설정입니다. 관절은 `slider_to_cart`(카트, 직선)와 `cart_to_pole`(막대, 회전) 두 개입니다.
- `{ENV_REGEX_NS}` 는 `/World/envs/env_0`, `env_1`, … 로 바뀝니다. 같은 로봇이 환경 수만큼 복사됩니다 (③ 페이지의 병렬 시뮬레이션).

### 행동 (Actions)

```python
@configclass
class ActionsCfg:
    # one number per env: force on the cart, action 1.0 = 100 N
    cart_force = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
```

- 정책의 출력은 숫자 하나입니다. `scale=100` 이라서 정책이 1.0 을 내면 카트에 100 N 의 힘이 걸립니다.
- 막대 관절은 행동에 없습니다. 막대는 카트의 움직임으로만 간접적으로 제어됩니다.

### 관측 (Observations)

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)    # cart position, pole angle (2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)    # cart velocity, pole angular velocity (2)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

- 정책이 보는 것은 카트 위치 · 막대 각도 · 카트 속도 · 막대 각속도, 숫자 4개입니다.
- `concatenate_terms=True` : 항목들을 이어 붙여 하나의 벡터(4차원)로 정책에 넣습니다.

### 이벤트 (Events): 리셋

```python
@configclass
class EventsCfg:
    reset_cart = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                           params={"asset_cfg": CART, "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)})
    reset_pole = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                           params={"asset_cfg": POLE, "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                                   "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)})
```

- 에피소드가 끝날 때마다(`mode="reset"`) 카트는 ±1 m, 막대는 ±45° 안에서 무작위로 시작합니다.
- 매번 같은 자세에서만 시작하면 그 자세만 잘하는 정책이 됩니다. 다양한 시작 상태가 곧 다양한 연습 문제입니다.

### 보상 (Rewards)

```python
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=pole_angle_sq, weight=-1.0, params={"asset_cfg": POLE})
    cart_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.01, params={"asset_cfg": CART})
    pole_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005, params={"asset_cfg": POLE})
```

![보상 항목과 가중치](l04_rewards.png){width=1000}

### 보상 함수 직접 쓰기

```python
def pole_angle_sq(env, asset_cfg):
    """Pole angle squared: 0 when upright, grows as it tips over."""
    angle = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)
```

- 보상 함수는 `env` 를 받아 **환경 수만큼의 숫자**(텐서 한 줄)를 돌려주는 파이썬 함수입니다. 4096개 환경을 한 번에 GPU에서 계산합니다.
- `joint_pos[:, ids]` : 모든 환경(`:`)의 막대 관절 각도. 이 값을 제곱하면 세워져 있을 때 0, 기울수록 커집니다. 가중치가 -1 이라 기울수록 감점입니다.

### 종료 (Terminations)

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out = DoneTerm(func=mdp.joint_pos_out_of_manual_limit, params={"asset_cfg": CART, "bounds": (-3.0, 3.0)})
```

- `time_out` : 5초(`episode_length_s`)가 지나면 끝. 실패가 아니라 "시간이 다 됨"이라서 `time_out=True` 로 구분합니다.
- `cart_out` : 카트가 레일 끝(±3 m)을 넘으면 실패로 끝납니다. 이때 `terminating` 보상(-2)을 받습니다.

### 환경 조립

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
```

### 시간 설정

```python
    def __post_init__(self):
        self.sim.dt = 1 / 120             # physics 120 Hz
        self.decimation = 2               # policy 60 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
```

- 물리는 1/120 초마다, 정책은 물리 2번에 한 번(60 Hz) 행동을 냅니다 (③ 페이지의 dt · decimation).
- 에피소드 5초 = 정책 300 스텝입니다. 학습 로그의 `Mean episode length` 가 300 에 가까우면 대부분 끝까지 버틴 것입니다.

## 3. 해 볼 것

- `ActionsCfg` 의 `scale` 을 10 으로 줄이면 카트를 밀 힘이 모자랍니다. ⑤ 페이지에서 학습해 보고 결과를 비교합니다.
- `EventsCfg` 의 막대 시작 범위를 ±45° 에서 ±5° 로 줄이면 학습은 쉬워지지만, 크게 기운 상태에서 시작하면 어떻게 될까요?
