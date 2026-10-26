# 외부 구현과 공개 범위

- 양자화기와 오프라인 추론 런타임은 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)의 **b9010**을 사용합니다. 설치 과정에서 원본 소스를 받습니다. MIT 라이선스와 저작권 표시는 원본 저장소에 있습니다.
- 학생 모델은 [Google Gemma](https://ai.google.dev/gemma)의 Gemma 3 270M 계열이며, 교사는 ECoG로 학습한 HyperCLOVA X 계열입니다. ECoG의 방법과 학습 코드는 [ECoG 저장소](https://github.com/kimsan1120/ECoG)를 참고하세요.
- 이 저장소는 양자화·평가 코드와 집계 결과를 제공합니다. 모델 가중치와 평가 데이터는 별도로 준비하며, 사용·공유 시 각 출처의 이용 조건을 따릅니다.
- 연구 당시 사용한 소스와 배포를 위해 정리한 실행 코드는 [코드 출처](docs/PROVENANCE.md)에 정리했습니다.
