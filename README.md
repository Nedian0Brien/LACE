# LACE

**LACE: Latent Adaptive Compression and Expansion for Language Diffusion**

이 저장소는 논문 주제인 *Compression, Not Corruption*의 핵심 주장, 즉 언어 확산 모델의 시간축을 임의 손상 스케줄이 아니라 정보율 경로로 다루어야 한다는 가설을 작은 규모에서 검증하기 위한 연구 스캐폴드다.

현재 실행 가능한 목표는 **Phase 0**이며, Kaggle에서 바로 돌릴 수 있는 최소 검증 실험이다.

1. 작은 텍스트 배치를 불러온다.
2. 고정된 `t5-small` encoder로 latent를 추출한다.
3. average pooling으로 단계별 latent shape를 만든다.
4. latent cache를 저장하고 다시 불러온다.
5. `metrics.json`과 `summary.md`를 기록한다.

## Kaggle에서 Phase 0 실행

```bash
kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status dennisparknd/lace-phase-0-latent-cache
kaggle kernels output dennisparknd/lace-phase-0-latent-cache -p outputs/phase0
```

Kaggle 스크립트는 독립 실행형으로 작성되어 있어 전체 저장소를 패키징하지 않아도 Kaggle 스크립트 커널에서 실행할 수 있다.

## 로컬 간단 검증

문법만 확인할 때는 로컬 환경에 GPU 의존성을 설치할 필요가 없다.

```bash
python3 -m py_compile kaggle/phase0/run_phase0.py
```

로컬에서 실제 encoder까지 실행하려면 먼저 PyTorch와 Transformers를 설치해야 한다.
