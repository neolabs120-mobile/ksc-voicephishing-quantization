# 결과 읽는 법

CSV와 JSON의 원천 파일·크기·SHA-256은 [`results/provenance.json`](../results/provenance.json)에 기록했습니다. 집계값을 추출한 자료이며, 원문이 포함된 실험 기록은 재배포하지 않습니다.

## 오프라인 전체 평가

| 증류 Gemma 구성 | 파일 MB | Test Macro-F1 | Challenge Macro-F1 | Challenge Recall | Challenge FPR |
|---|---:|---:|---:|---:|---:|
| F32 GGUF | 1,078.92 | 99.00% | 95.59% | 97.47% | 6.23% |
| 전 블록 Q8_0 / E8 | 291.55 | 99.15% | 95.81% | 97.62% | 5.94% |
| 전 블록 Q4_0 / E8 | 241.41 | 94.64% | 89.21% | 79.28% | 0.87% |
| B03·B05·B11 Q8_0 / 나머지 Q4_0 / E8 | 249.77 | 98.15% | 95.00% | 94.78% | 4.78% |

Test는 712건, Challenge는 1,361건입니다. 소수점 이하를 반올림했으며 정확한 값과 혼동행렬은 CSV에 있습니다. Q4/Q8 대조군도 공유 임베딩·출력 헤드는 Q8_0이고 정규화는 F32입니다. F32 기준은 임베딩까지 F32입니다.

| 파일 | 내용 |
|---|---|
| [`desktop_baselines.csv`](../results/desktop_baselines.csv) | 증류 전·후 F32와 전 블록 Q4/Q8, 분할별 혼동행렬 |
| [`teacher_baselines.csv`](../results/teacher_baselines.csv) | 원본·ECoG 교사 비교와 문맥 초과 제외 건수 |
| [`single_block_validation.csv`](../results/single_block_validation.csv) | Q4 대조군과 단일 Q8 블록 18개의 Validation 성능 |
| [`cumulative_candidates.csv`](../results/cumulative_candidates.csv) | 민감도 순서로 Q8 블록을 누적한 17개 조합의 Test·Challenge 성능 |
| [`final_model.csv`](../results/final_model.csv) | 최종 모델의 지표·혼동행렬·기술적 Wilson 구간 |
| [`paper_table1.csv`](../results/paper_table1.csv) | 논문 표 1의 반올림된 수치 |
| [`model_layouts.json`](../results/model_layouts.json) | 실제 저장 텐서 형식, 크기, 모델 해시 |

Validation은 학습과 공백 정규화 후 중복된 24건을 제외한 692건입니다. 단일 블록의 Recall, Macro-F1, FPR 순으로 우선순위를 정한 뒤 누적 조합을 비교했습니다. 최종 선택은 관측된 Challenge FPR 5% 이하 조합 중 Recall을 우선했습니다. Test·Challenge 재사용과 Train/Test 중복 10건이 있으므로 독립적인 일반화 성능을 확정하는 자료는 아닙니다. Wilson 구간도 후보 선택 편향을 보정하지 않습니다.

판정은 첫 응답 위치의 전체 어휘 argmax가 `0/1`이면 그대로 사용하고, 다른 토큰이면 두 라벨 logit을 비교합니다. `native_invalid_first_token_count`는 이때 라벨 외 토큰이 argmax였던 건수입니다. 최종 학생의 두 분할에서는 0건입니다. 증류 전 Gemma는 모든 입력에서 자연 첫 토큰이 라벨이 아니었으므로, 해당 수치는 **강제 0/1 선택에 따른 보조 결과**입니다. 모델의 일반적인 언어 능력을 뜻하지 않습니다.

교사 기준 결과는 학생 GGUF와 실행 경로가 다릅니다. 교사의 Challenge는 문맥 한도로 4건을 제외한 1,357건이며, 저장 정밀도와 계산 정밀도도 구분합니다. `teacher_baselines.csv`의 평가 분모와 판정 규칙을 함께 확인해야 합니다.

오프라인 참조 조건은 M4 Pro·Metal, llama.cpp b9010, CPU 4스레드, 문맥 32,768, 배치 512, F32 KV입니다. 전체 전사에 대한 판정 지표이며 ASR·설명 생성·경고 표시까지의 지연은 포함하지 않습니다.

## 스마트폰 측정

| 배포 모델 | 파일 MB | 판정 중앙값 ms | 판정 P90 ms | 앱 PSS 관찰 최대 MiB | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| ECoG 교사 Q3_K / E8 | 319.23 | 8,521.1 | 21,904.2 | 1,537.2 | 97.22% |
| 최종 혼합 학생 | 249.77 | 799.7 | 2,168.7 | 995.0 | 97.22% |

[`android_measurements.csv`](../results/android_measurements.csv)는 **논문에 기록된 측정 집계**입니다. 이 공개본에 원본 모바일 결과 ZIP은 없으므로 `evidence_level=paper_reported`로 구분했습니다. 단말명·OS는 논문의 기재값을 보존했습니다.

Challenge의 정부·금융, 정상·피싱, 길이 구간을 고려한 36건 표본에서 각 모델을 측정 모드별 1회 평가했습니다. CPU 8스레드, 문맥 8,192, F16 KV, 배치 512이며 로드마다 준비 실행 2회를 제외했습니다. 속도와 메모리는 별도로 측정했습니다.

판정 시간은 토큰화·입력 처리·0/1 점수 비교를 포함하고 모델 로딩과 설명 생성을 제외합니다. P90은 관측값의 90백분위수입니다. PSS는 판정 전후와 1초 간격으로 관측한 **앱 전체 메모리의 최대치**이며, 모델 가중치 크기나 연속 측정한 절대 peak를 뜻하지 않습니다.

교사와 학생은 구조·토크나이저·양자화 형식·파일 크기가 다릅니다. 이 결과는 두 배포 구성의 비교이며 차이를 증류 하나의 효과로 분리해 해석하지 않습니다.

## 교사 양자화와 제외한 실험

교사 배포 모델은 `Q3_K_S`와 공유 임베딩·출력 Q8_0을 사용했습니다. F32 교사 GGUF를 별도로 준비한 경우의 명령은 다음과 같습니다. 학생 전용 `inspect()`는 18블록 Gemma 구조를 검사하므로 교사에 적용하지 않습니다.

```bash
build/bin/llama-quantize --token-embedding-type q8_0 --output-tensor-type q8_0 \
  work/teacher_f32.gguf work/teacher_q3ks_e8.gguf Q3_K_S 4
```

학생 Q3_K 실험은 선형 행렬 126개 중 90개가 Q4_0으로 저장되어 ‘전 블록 3비트’ 비교로 사용하지 않았습니다. 과거 별도로 수행한 GPTQ 결과도 이 저장소의 최종 GGUF 결과와 합치지 않습니다.
