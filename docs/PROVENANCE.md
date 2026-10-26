# 코드 출처와 공개용 정리

KSC 논문에 사용한 GGUF 양자화·평가 코드와 집계 결과를 공개용으로 정리했습니다. 실행 경로는 인자로 지정할 수 있도록 바꿨습니다.

## 연구 코드

원 작업 폴더의 `decoder_distillation/scripts/`를 기준으로 다음 코드를 가져왔습니다.

| 공개 파일 | 역할 | 정리 범위 |
|---|---|---|
| `scripts/prepare_mobile_gguf.py` | F32 변환, 혼합 Q4/Q8 명령, 실제 저장 텐서 검사 | 도움말을 의존성 설치 전에 볼 수 있도록 `gguf` import만 지연 |
| `scripts/evaluate_mobile_gguf.py` | 토크나이저 일치 입력 준비, HF 기준값, 집계 | 바이트 동일 |
| `scripts/gguf_eval.cpp` | llama.cpp C API 추론, 첫 응답 위치 logit, 선택적 설명 생성 | 바이트 동일 |
| `scripts/common/` | CSV·JSONL 읽기, 해시, 지표, 학생 응답 계약 | 포함한 4개 파일은 바이트 동일 |
| `prompts/student_system_prompt.txt` | 학생 학습·평가의 고정 지시문 | 원본 체크포인트의 지시문과 바이트 동일 |

파일별 원천 상대경로와 원본·공개본 SHA-256은 [`SOURCE_MANIFEST.json`](../SOURCE_MANIFEST.json)에 있습니다. 위 파일은 연구 중 작성한 실행 코드이며, 외부 구현인 llama.cpp와 ECoG는 각각의 원 출처를 따릅니다.

## 공개용 연결 코드

`scripts/public_pipeline.py`, `CMakeLists.txt`, `tests/test_pipeline.py`는 공개용으로 정리하면서 구성했습니다. 기존 변환·추론·지표 함수를 사용하되, 개인 컴퓨터의 이전 실험 폴더가 없어도 경로를 인자로 지정할 수 있게 연결했습니다.

단일 블록 비교와 누적 비교는 기존 `run_gguf_block_sweep.py`, `run_gguf_cumulative.py`의 절차를 옮겼습니다. 각 실행의 Validation 결과로 Recall 내림차순, Macro-F1 내림차순, FPR 오름차순, 블록 번호 오름차순의 순위를 계산합니다. 누적 후보는 2~18블록의 17개입니다.

원래 누적 실행기는 후보 표를 저장했습니다. 공개본에는 논문의 선정 기준에 따라 FPR 조건을 만족하는 최대 Recall 후보를 기록하는 단계를 추가했습니다. 동률 후보는 모두 남깁니다.

`results/`에는 연구 당시 실험 결과를 집계했으며, 실행 서명과 예측도 당시 기록을 기준으로 유지했습니다. 공개용 연결 코드는 아래 빌드와 테스트로 검증했습니다.

## 런타임과 결과 출처

- 오프라인 GGUF 변환·평가: llama.cpp **b9010**, 커밋 `d05fe1d7dadbf8943c8f1903fcf65b935ddab839`.
- Android 측정 런타임: 커밋 `5f55650a78f92aff4d48d671423e888fac0469ff`.
- 집계 결과: [`results/provenance.json`](../results/provenance.json)에 원천 15개 파일과 논문 PDF의 해시를 기록했습니다. 공개 자료에는 개인정보를 제외한 집계값을 실었습니다.
- 공개본 검증 범위: 고정 패키지 설치, 양자화기·평가기 빌드, 명령·텐서 배정·입력 대응·지표 테스트.

## 포함하지 않은 자료

Teacher 학습 원본, 응답 증류 학습기, 체크포인트, 통화 CSV, 개별 예측·응답·입력 토큰, 서버 접속 정보, 인증정보, RAG·앱 통합 코드는 공개 대상에서 제외했습니다. 재실험에 필요한 모델과 데이터는 별도로 준비해야 하며, 사용·공유 조건은 각각의 원 출처를 따릅니다.
