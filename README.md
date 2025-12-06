# Causal Reinforcement Learning for Bandits Implementation

📌 본 저장소는 논문 "Causal Reinforcement Learning for Bandits with Unobserved Confounders" (Mingwei Deng, 2023)의 알고리즘을 Python으로 직접 구현하고 재현한 프로젝트입니다.

단순 재현을 넘어, 논문에서 고려하지 않은 새로운 환경 변수(교란 강도 $\alpha$, 비선형성 $\lambda$)**를 추가하여 알고리즘의 강건성(Robustness)을 검증하는 확장 실험 코드를 포함하고 있습니다.

---

### 📂 주요 기능 및 알고리즘

1.  알고리즘 1: Binary Causal Bandit
    * 이진 변수 환경($Z, X, Y \in \{0, 1\}$)에서의 인과 밴딧 알고리즘 구현.
    * `CausalAgent`, `CUCBAgent`, `CTSAgent` 성능 비교.
    * [확장 기능] 교란 강도 파라미터 `alpha_conf` 추가: 소스/타겟 도메인 간의 인과 구조 세기 차이를 시뮬레이션.

2.  알고리즘 2: Continuous (VAE) Causal Bandit
    * 연속형 변수 및 고차원 프록시 환경에서의 VAE 기반 인과 밴딧 구현.
    * `CausalVAEAgent` (Encoder-Decoder 구조) vs `LinUCBAgent` 성능 비교.
    * **[확장 기능]** 비선형 강도 파라미터 `NONLINEAR_STRENGTH` 추가: 데이터 생성 과정에 비선형 항($\sin(Z)$)을 주입하여 모델 불일치(Model Mismatch) 환경 테스트.

---

### 🚀 실행 방법 (Usage)

#### 1. 필수 라이브러리 설치
본 코드는 `tensorflow`, `numpy`, `matplotlib`, `scipy` 등을 사용합니다. 한글 폰트 지원을 위해 `koreanize-matplotlib`도 필요합니다.

```bash
pip install tensorflow numpy matplotlib scipy koreanize-matplotlib
