# KSC 2026 · 보이스피싱 탐지 모델 양자화

**응답 기반 증류와 혼합 정밀도 양자화를 이용한 보이스피싱 탐지 모델 경량화**의 양자화·평가 코드와 실험 집계 결과입니다.

ECoG 교사의 판정·설명 응답을 학습한 Gemma 3 270M을 F32 GGUF로 변환한 뒤, llama.cpp의 Q4_0/Q8_0으로 양자화했습니다. GGUF 변환부터 블록별 정밀도 비교, 탐지 성능 평가까지의 코드와 결과를 모았습니다.

## 최종 구성

| 가중치 | 저장 형식 |
|---|---|
| B03·B05·B11의 Attention·FFN 선형 가중치 | Q8_0 |
| 나머지 15개 블록의 선형 가중치 | Q4_0 |
| 입력 임베딩과 이를 공유하는 출력 헤드 | Q8_0 |
| 정규화 가중치 | F32 |

블록 번호는 **0부터** 시작합니다. 126개 선형 가중치 중 21개가 Q8_0, 105개가 Q4_0입니다. 공유 임베딩을 포함한 Q8_0 텐서는 22개입니다. 양자화는 가중치에 적용하며, KV 캐시 형식은 실행 옵션으로 지정합니다.

최종 파일은 **249,766,048 bytes = 249.77 MB**입니다. 파일 크기의 MB는 10⁶ bytes이며, 실행 메모리의 MiB와 구분합니다.

## 코드와 결과

```text
scripts/
  prepare_mobile_gguf.py   연구에 사용한 변환·혼합 양자화·텐서 검사
  evaluate_mobile_gguf.py  평가 입력 준비와 지표 집계
  gguf_eval.cpp           llama.cpp C API로 첫 토큰 점수 계산
  common/                데이터·응답 형식·지표 공통 코드
  public_pipeline.py     공개용 실행 연결부: 변환·양자화·평가·블록/누적 탐색
results/                 기존 실험에서 추출한 집계 수치와 출처
docs/                    실험 조건·공개 범위·코드 출처
tests/                   명령 구성·입력 검증·지표 검사
```

`results/`에 논문 실험의 집계 결과를 저장했습니다. 원천 파일 경로와 해시는 [출처 문서](docs/PROVENANCE.md)와 결과 기록에 있습니다.

## 실행 준비

Python 3.10~3.13, CMake 3.20 이상, C++17 컴파일러, Git이 필요합니다. 패키지 설치는 Python 3.13에서 확인했습니다. CUDA 빌드에는 CUDA 개발 도구도 필요합니다. 변환 입력은 **증류 완료·양자화 전 Gemma 3 270M Hugging Face 체크포인트**입니다.

체크포인트와 평가 CSV는 별도로 준비합니다. 체크포인트에는 토크나이저·chat template·`student_system_prompt.txt`가 함께 있어야 합니다. 학습 지시문은 [`prompts/student_system_prompt.txt`](prompts/student_system_prompt.txt)에 있습니다.

```bash
git clone https://github.com/neolabs120-mobile/ksc-voicephishing-quantization.git
cd ksc-voicephishing-quantization
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

git clone --branch b9010 --depth 1 https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp rev-parse HEAD
# 고정 커밋: d05fe1d7dadbf8943c8f1903fcf65b935ddab839
python -m pip install ./third_party/llama.cpp/gguf-py

cmake -S . -B build -DLLAMA_CPP_DIR="$PWD/third_party/llama.cpp" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build build --target llama-quantize gguf_eval --parallel 4
```

위 명령은 CPU 빌드입니다. CUDA를 사용할 때는 CMake 명령에 `-DGGML_CUDA=ON`, 아래 평가 명령에 `--gpu-layers 99`를 추가합니다. Apple Metal을 사용할 때는 `-DGGML_METAL=ON`으로 바꾸고 `--gpu-layers 99`를 사용합니다. 논문 오프라인 참조 결과는 Metal 환경에서 얻었으므로 다른 백엔드의 수치 일치는 별도로 확인해야 합니다.

## 변환과 양자화

경로의 `work/checkpoints/student`는 직접 준비한 **증류 완료·양자화 전** 체크포인트입니다.

```bash
python scripts/public_pipeline.py convert \
  --source work/checkpoints/student --llama-dir third_party/llama.cpp --output work/f32

# 최종 혼합 구성
python scripts/public_pipeline.py quantize \
  --f32 work/f32/Student_F32.gguf --quantizer build/bin/llama-quantize \
  --precision mixed --q8-blocks 3 5 11 --output work/final.gguf

# 전 블록 Q4_0 대조군. 공유 임베딩·출력 헤드는 Q8_0 유지.
python scripts/public_pipeline.py quantize \
  --f32 work/f32/Student_F32.gguf --quantizer build/bin/llama-quantize \
  --precision q4 --output work/q4_e8.gguf

# 전 블록 Q8_0 대조군. 정규화는 F32 유지.
python scripts/public_pipeline.py quantize \
  --f32 work/f32/Student_F32.gguf --quantizer build/bin/llama-quantize \
  --precision q8 --output work/q8_e8.gguf
```

양자화 후 저장 텐서의 형식·파일 크기·SHA-256을 확인합니다. ‘전 블록 Q4/Q8’은 Transformer **선형 가중치**의 정밀도를 가리킵니다. 임베딩과 정규화의 형식은 위 구성표를 따릅니다.

## 평가와 블록별 비교

CSV 폴더에는 `E_train.csv`, `E_validation.csv`, `E_test.csv`, `E_challenge.csv`를 준비합니다. 기본 열은 `text`, `label`, `id`이며 정상은 `0`, 보이스피싱은 `1`입니다.

```bash
python scripts/public_pipeline.py prepare-inputs \
  --source work/checkpoints/student --data work/data --output work/inputs

python scripts/public_pipeline.py evaluate \
  --model work/final.gguf --inputs work/inputs/evaluation.inputs.jsonl \
  --runner build/bin/gguf_eval --output work/final_eval --gpu-layers 0 --kv f32

python scripts/public_pipeline.py sweep \
  --f32 work/f32/Student_F32.gguf --inputs work/inputs/validation.inputs.jsonl \
  --quantizer build/bin/llama-quantize --runner build/bin/gguf_eval --output work/block_sweep

python scripts/public_pipeline.py cumulative \
  --f32 work/f32/Student_F32.gguf --validation-results work/block_sweep/validation.json \
  --inputs work/inputs/evaluation.inputs.jsonl --quantizer build/bin/llama-quantize \
  --runner build/bin/gguf_eval --output work/cumulative --max-fpr 0.05

python -m unittest discover -s tests -v
```

평가기는 Hugging Face와 GGUF의 입력 토큰이 같은지 확인합니다. 첫 토큰 전체 어휘 argmax가 `0/1`이면 그대로 사용하고, 다른 토큰이면 `0/1`의 logit을 비교합니다. 이 판정을 기준으로 탐지 성능을 계산하며, 라벨 외 첫 토큰이 나온 건수도 함께 기록합니다.

학습과 중복된 Validation 통화를 제외하고 단일 블록을 비교한 뒤, Q8 블록을 누적한 17개 조합을 평가했습니다. 최종 구성은 Challenge FPR 5% 이하 후보 중 Recall을 우선하여 선택했습니다. **Test·Challenge도 후보 비교에 반복 사용했으며, 보고한 수치는 탐색 과정에서 관측한 성능입니다.**

스마트폰에서는 별도로 선별한 36건을 측정했습니다. PSS는 **앱 전체에서 관찰한 메모리 최댓값**입니다. 세부 조건은 [결과 안내](docs/RESULTS.md)에 있습니다.

외부 구현과 모델·데이터의 출처 및 사용 범위는 [NOTICE](NOTICE.md)를 참고하세요.
