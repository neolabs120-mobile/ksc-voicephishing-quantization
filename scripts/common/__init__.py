"""학습·양자화·평가가 공유하는 순수 Python 계약. 모델·장치 라이브러리는 여기서 적재하지 않는다."""

from pathlib import Path

# 실행 서명에 호환 파일뿐 아니라 실제 구현도 넣어, 공통 코드 변경이 재개 검사에서 빠지지 않게 한다.
SOURCE_FILES = tuple(f"common/{p.name}" for p in sorted(Path(__file__).parent.glob("*.py")))
