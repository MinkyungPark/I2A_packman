# I2A Architecture
- keywords : data efficiency, Imperfection model(e.g. f-approx) robustness, policy distillation
- Training $\pi, V$ : policy gradient, value function approximation

<br>

![](https://t1.daumcdn.net/cfile/tistory/99BF2C3E5B0CFFC626)

<br>

a) Imagination core <br>
- $\hat{\pi}$ Simulation ot -> ot+1, rt+1, ot+2, rt+2...
- rollout strategy : Policy distillation
  - pretrained policy(imperfect model) : $\hat{\pi}(o_t)$
  - imagination-augmented policy : $\pi(o_t)$
  - $L_{dist}(\pi, \hat{\pi}) = λ_{dist}Σ_aπ(a|o)log(\hat{\pi}(a|o))$
- Policy Network
  - 랜덤, pretrained model ... 등등 여기서는 a2c를 사용
  - out, n개의 trajectories  $\hat{τ}_1,...\hat{τ}_n$
- Environment Model
  - distribution which is optimized by using NLL($l_{model} target Reward,State와 imagined Reward,state와의 negative likelihood loss$)
  - 환경 모델의 dynamic이랑 env_model의 dynamics랑 $l_{model}$을 통해 맞춘다.
  - Learning Env dynamics from training data from partial trained model-free agent(Policy Network) trajectories(~$\hat{\pi}$)

b) Encode <br>
- ⭐️ Rollout encoding features is out encoded possible trajectories
    - rollout embedding $e_i \ = \ ϵ(\hat{Τ}_i)$

    - Aggregator out singloe imagination code $c_{ia}$ = 𝛢$(e_1, e_2, .... e_n)$

- Imagination-augmented RL : Learn to Interpret <br>
  - model free decision을 더 효율적으로 하기 위해 action 전, simulation을 한다.
  - immediate reward와는 상관 없어도 $τ$ 확인시 value와 관련 높은 정보를 추출하거나 필요 없는 $τ$ 무시 한다!
  - 즉, imperfection model의 useful but imperfect simulation knowledge를 추출해 end-to-end로 학습함 <br>

c) Out <br>
- whole rollout encoding $c_ia$ + model-free path $c_mf$ => $\pi, V$

<br>

---
## Details
### a) Imagination Core(IC)
### 1. Policy Network A2C
a2c.py a2c_train.ipynb
- it can reuse Model-free path
- out, n개의 trajectories $\hat{τ}_1, ...\hat{τ}_n$

### 2. Env Model
env_model.py env_model_train.ipynb
<br>

<a href='https://ifh.cc/v-69TTc2' target='_blank'><img src='https://ifh.cc/g/69TTc2.png' border='0'></a>
- Env Model training : predict next frame and reward by stochastic gradient descent on the bernoulli cross-entropy(negative log-likelihood loss $l_{model}$)
- Reward Vector $w_{rew} \in R^5$ = moving/ eating food/ eating a power pill/ eating a ghost/ being eaten by a ghost
  - reward is sparse
  - prediction of ghost dynamics is important

### b) Imagination Rollout Encoder
encoder.py

<br>

---

<br>

### Env : MiniPacman
- state space : (3, 15, 19) -> (RGB X WIDTH X HEIGHT)
- actions : Discrete(5).  [up, left, right, down, stay]
- observation_space : spaces.Box(low=0, high=1.0, shape=(3, 15, 19))
- modes : regular, avoid, hunt, ambush, rush
- instance : MiniPacman(mode, frame_cap)

| Environment | Regular | Avoid | Hunt | Ambush | Rush |
| :----------- | :-----------: | :-----------: | :-----------------: | :-----------------: | :----------: |
| Step Reward       | 0 | 0.1 | 0 | 0 | 0 |
| Food Reward       | 1 | -0.1 | 0 | -0.1 | -0.1 |
| Power Pill Reward | 2 | -5 | 1 | 0 | -10 |
| Kill Ghost Reward | 5 | -10 | 10 | 10 | 0 |
| Death Reward      | 0 | -20 | -20 | -20 | 0 |
| Next level if all power pills eaten | No | No | No | No | Yes |
| Next level if all ghosts killed | No | No | Yes | Yes | No |
| Next level if all food eaten | Yes | Yes | No | No | No |
| Next level when surviving n timesteps | Never | 128 | 80 | 80 | Never |

<br>

references
- https://github.com/mpSchrader/gym-sokoban/issues/11
- [TEMPORAL DIFFERENCE MODELS: MODEL-FREE DEEP RL FOR MODEL-BASED CONTROL](https://arxiv.org/pdf/1802.09081.pdf)
- [Learning model-based planning from scratch](https://arxiv.org/pdf/1707.06170.pdf)

