# 실험 결과 요약

이 디렉터리는 LACE 연구에서 실제로 실행한 실험 결과를 정리하는 공간이다. 단순히 숫자를 보관하는 것이 아니라, 각 실험이 무엇을 검증했는지, 어떤 지표를 어떻게 해석해야 하는지, 다음 실험으로 무엇을 넘겨야 하는지를 함께 기록한다.

Kaggle을 사용해 새 실험을 계획하거나 실행할 때는 먼저 [kaggle-experiment-workflow.md](./kaggle-experiment-workflow.md)를 확인한다. 이 문서는 runner 구성, 로컬 검증, Kaggle push/status/output, 결과 문서화, 커밋/푸쉬 마무리 절차를 정리한 운영 문서다.

현재까지 실행한 주요 실험은 다음과 같다.

| 실험 | 문서 | 상태 | 한 줄 결론 |
|---|---|---|---|
| Phase 0 | [phase-0-latent-cache.md](./phase-0-latent-cache.md) | 완료 | Kaggle GPU에서 frozen T5 encoder latent cache와 단계별 pooling metric pipeline이 정상 동작했다. |
| Phase 1 smoke | [phase-1-latent-compression-smoke.md](./phase-1-latent-compression-smoke.md) | 완료 | Fallback corpus에서 reverse expander 학습 가능성은 확인했지만, learned attention compression은 아직 latent 사용 신호가 약했다. |
| Phase 1 WikiText-2 | [phase-1-wikitext-2-subset.md](./phase-1-wikitext-2-subset.md) | 완료 | 실제 WikiText-2 subset에서는 문제가 더 선명해졌다. Pooling은 학습되지만 latent sensitivity가 낮고, attention compression은 아직 구조 개선이 필요하다. |
| Phase 2 forward isolation | [phase-2-forward-process-isolation.md](./phase-2-forward-process-isolation.md) | 완료 | Average pooling compression은 MSE 우위는 보이지 못했지만, random/noise baseline보다 cosine과 token reconstruction bridge에서 좋은 초기 신호를 보였다. |
| Phase 2B calibrated validation | [phase-2b-calibrated-validation.md](./phase-2b-calibrated-validation.md) | 완료 | Matched Gaussian과 2048-sample run에서도 average pooling은 frozen decoder NLL과 latent-use 신호를 유지했지만, MSE/cosine/token-head/open-ended generation은 강하게 통과하지 못했다. |
| Phase 2C positional encoding | [phase-2c-positional-encoding.md](./phase-2c-positional-encoding.md) | 완료 | Explicit sinusoidal positional conditioning은 reverse expansion proxy를 크게 개선했지만, position-only/Gaussian positional control도 강해 Average Pooling 고유 우위는 아직 미입증이다. |
| Phase 2D positional confound bridge | [phase-2d-positional-confound-bridge.md](./phase-2d-positional-confound-bridge.md) | 완료 | Absolute/relative position을 분리했고, `average_pool_rel_pos`를 Phase 3A primary condition으로 삼되 position-only/Gaussian controls를 반드시 유지해야 한다는 결론을 얻었다. |
| Phase 3A generation-aware reverse objective | [phase-3a-generation-aware-reverse-objective.md](./phase-3a-generation-aware-reverse-objective.md) | 완료 | Frozen token-head loss는 token-head NLL을 크게 낮췄지만 decoder NLL과 open-ended generation을 악화시켜, 현재 proxy objective가 decoder-compatible generation objective가 아님을 보였다. |

## 현재 연구 상태

현재 결과는 **초기 검증 단계**다. 연구 가설을 증명한 결과는 아니지만, 다음 두 가지는 확인했다.

1. `t5-small` encoder로 만든 latent `h0`를 Kaggle GPU에서 안정적으로 cache할 수 있다.
2. 압축된 latent `z_t`에서 더 상세한 latent 또는 `h0`를 복원하도록 expander를 학습할 수 있다.

이후 WikiText-2 subset으로 Phase 1을 다시 실행했다. 실제 데이터에서는 fallback corpus보다 복원 난도가 올라갔고, 특히 learned attention compression의 약점이 더 분명해졌다. 현재 결과의 올바른 해석은 다음과 같다.

> 실험 파이프라인과 reverse expansion 학습 루프는 살아 있다. 그러나 learned compression의 우수성은 아직 입증되지 않았고, compression forward의 장점도 hidden reconstruction만으로는 강하게 말할 수 없다.

Phase 2B에서는 Gaussian 난도 보정, sample/epoch 확장, h0 decoder control, lightweight token-head bridge를 추가했다. 그 결과 average pooling은 frozen decoder delta NLL과 latent-use에서는 살아 있는 신호를 보였지만, MSE/cosine/token-head/open-ended generation에서는 엄격한 통과를 얻지 못했다.

Phase 2C에서는 expander에 fixed sinusoidal positional feature를 추가했다. Average Pooling positional은 baseline Average Pooling보다 MSE, cosine, decoder NLL, token-head NLL을 모두 크게 개선했다. 그러나 position-only control도 강했고, Gaussian positional은 더 강한 MSE/cosine/token/generation 신호를 보였다. 따라서 positional scaffold는 중요한 요소로 확인됐지만, 이것이 Average Pooling compression의 고유한 장점이라는 결론은 아직 보류해야 한다.

Phase 2D에서는 positional confound를 Phase 3A로 넘어가기 위한 bridge로 정리했다. Absolute, block-relative, absolute+relative positional mode를 분리했고, shuffled-label delta와 wrong-position sweep을 추가했다. 결과적으로 Average Pooling positional은 position-only보다 cosine, frozen decoder NLL, shuffled-label sensitivity에서 content contribution을 보였지만, Gaussian absolute positional이 여전히 가장 강한 반론으로 남았다. Phase 3A의 primary condition은 `average_pool_rel_pos`가 가장 해석 가능하며, `position_only_rel_pos`, `position_only_abs_rel_pos`, `gaussian_noise_abs_pos`를 필수 control로 유지해야 한다.

Phase 3A에서는 frozen token-head loss를 reverse expander 학습 objective에 직접 추가했다. Token-head delta NLL은 크게 좋아졌고 일부 arm에서는 원본 `h0`보다 token-head에 더 잘 맞는 음수 delta까지 나왔다. 그러나 MSE/cosine은 악화됐고, frozen decoder delta NLL도 나빠졌으며, Average Pooling token arms의 meaningful generation은 모두 0으로 유지됐다. Gaussian reconstruction-only control만 meaningful generation 0.416667을 보였고, Gaussian에 token loss를 넣으면 generation이 0으로 붕괴했다. 따라서 현재 lightweight token-head objective는 decoder-compatible generation objective가 아니라 proxy over-optimization 위험이 큰 objective로 보는 것이 타당하다.

## 지표를 읽는 법

| 지표 | 쉬운 의미 | 좋은 방향 |
|---|---|---|
| MSE | 복원한 latent가 원래 latent와 얼마나 다른지 보는 오차 | 낮을수록 좋음 |
| Cosine | latent 벡터 방향이 얼마나 비슷한지 보는 유사도 | 높을수록 좋음 |
| Validation loss | 학습에 직접 보지 않은 데이터에서의 복원 손실 | 감소해야 함 |
| Variance ratio | 복원 latent가 납작하게 collapse되지 않았는지 보는 비율 | 너무 낮으면 위험 |
| Perturbation sensitivity | latent를 살짝 흔들었을 때 출력도 변하는지 보는 검사 | 0에 가까우면 latent를 안 쓸 가능성 |
| Gate | 실험 단계 통과 여부를 판단하는 체크포인트 | pass가 많을수록 다음 단계 가능 |

## 현재까지의 핵심 판단

### 통과한 것

- Kaggle GPU 실행
- frozen `t5-small` encoder latent 추출
- latent cache 저장 및 재로딩
- 단계별 token budget 축소
- pooling 기반 compression에서 자연스러운 `z1 → z2 → z3` 난도 곡선
- pooling latent에서 `h0`를 복원하는 reverse expander 학습
- matched Gaussian calibration 기반 forward-process 비교
- frozen decoder bridge에서 average pooling의 delta token NLL 우위
- average pooling latent-use 신호

### 아직 약한 것

- learned attention compression이 pooling baseline보다 안정적이지 않다.
- attention compression의 perturbation sensitivity가 낮다.
- WikiText-2에서 pooling도 latent sensitivity gate를 통과하지 못했다.
- attention compression은 WikiText-2에서 stage 간 차이가 너무 작고 cosine이 낮다.
- 의미 보존 metric은 아직 측정하지 않았다.
- token-head bridge에서는 Gaussian이 average pooling보다 강하다.
- reconstructed `h0_hat` 기반 open-ended generation은 아직 붕괴되어 있다.
- positional conditioning은 매우 강하지만, position prior와 content latent contribution이 아직 충분히 분리되지 않았다.
- Gaussian positional이 Average Pooling positional보다 token/generation proxy에서 더 강하다.
- Phase 2D에서 content contribution 신호는 확인됐지만, token-head와 open-ended generation에서는 Gaussian positional이 여전히 더 강하다.
- Phase 3A에서 token-head objective는 token-head proxy를 과하게 최적화하면서 frozen decoder manifold와 open-ended generation을 악화시켰다.

## 다음 실험 우선순위

1. Phase 3B에서 token-head proxy 대신 decoder-compatible objective를 검토한다.
2. 우선 compact set으로 `average_pool_rel_pos_recon`, decoder-loss arm, `position_only_rel_pos`, `gaussian_noise_abs_pos`를 비교한다.
3. Frozen decoder NLL을 훈련 loss에 직접 넣거나, token-head loss가 hidden manifold를 벗어나지 못하도록 강한 hidden/cosine regularization을 둔다.
4. open-ended generation은 원본 `h0` control과 Gaussian reconstruction-only control을 항상 같이 비교한다.
5. Phase 3B 이후 여러 seed로 Phase 2D/3A/3B 핵심 지표를 재현한다.
