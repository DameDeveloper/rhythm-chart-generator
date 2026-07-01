# Rhythm Chart Generator

유튜브 링크 또는 로컬 WAV 파일을 입력하면 음악을 분석해서 리듬게임용 채보 JSON/CSV를 생성하는 Python 도구입니다.

## 특징

- BPM과 비트 오프셋 추정
- 온셋, 음량, 구간 에너지 기반 노트 배치
- 4/5/6/7/8키 채보 생성
- easy, normal, hard, expert 난이도 프리셋
- 탭, 롱노트, 제한적인 동시치기 생성
- 반복 타건과 무의미한 밀집을 줄이는 레인 패턴 규칙

특정 채보 제작자의 개별 채보를 복제하지는 않습니다. 대신 이지투, DJMAX, 프로젝트 세카이 같은 리듬게임에서 흔히 좋은 채보로 느껴지는 일반 원칙을 규칙화했습니다.

## 데스크톱 앱 (Windows exe)

설치 없이 바로 쓰는 단일 실행파일입니다. 실행하면 로컬 서버가 켜지고 **주소창 없는
웹앱 전용 창**(Edge/Chrome `--app` 모드, 없으면 기본 브라우저)으로 화면이 열립니다.
Python·ffmpeg·Node 를 설치할 필요가 없으며 ffmpeg 는 실행파일에 포함되어 있습니다.

### 방법 1 — 릴리스에서 내려받기 (가장 간단)

1. 저장소 우측 **Releases** → 최신 버전에서 `RhythmChartGenerator.exe` 다운로드
   (또는 CLI: `gh release download -R DameDeveloper/rhythm-chart-generator -p "*.exe"`)
2. 더블클릭해서 실행. 종료는 검은 콘솔 창을 닫으면 됩니다.
3. SmartScreen 경고가 뜨면 **추가 정보 → 실행** (서명되지 않은 파일이라 뜨는 정상 경고).

### 방법 2 — 소스에서 clone 후 실행/빌드

```powershell
git clone https://github.com/DameDeveloper/rhythm-chart-generator.git
cd rhythm-chart-generator

# (A) 빌드 없이 개발 모드로 바로 실행 (Python 필요)
python -m pip install -r backend\requirements.txt
python backend\desktop_app.py

# (B) 단일 exe 새로 빌드 (프론트 빌드 + PyInstaller 자동)
build_exe.bat
# 결과물: dist_exe\RhythmChartGenerator.exe
```

> exe 는 용량이 커서 git 저장소가 아니라 **GitHub Releases 자산**으로 배포합니다.
> 소스에는 빌드에 필요한 모든 것(런처·스펙·아이콘)이 들어 있어 `build_exe.bat`
> 한 번으로 언제든 다시 만들 수 있습니다.

아래는 명령줄(`chartgen.py`)로 직접 쓰는 방법입니다.

## 준비

기본 분석에는 Python과 NumPy가 필요합니다.

유튜브 링크를 바로 쓰려면 별도로 `yt-dlp`와 `ffmpeg`가 설치되어 있어야 합니다.

```powershell
python -m pip install numpy yt-dlp
```

`ffmpeg`는 Windows 패키지 매니저나 공식 배포본으로 설치한 뒤 PATH에 추가하세요.

## 사용법

로컬 WAV 파일:

```powershell
python chartgen.py "song.wav" --keys 4 --difficulty hard --out chart.json --csv chart.csv
```

유튜브 링크:

```powershell
python chartgen.py "https://www.youtube.com/watch?v=..." --keys 6 --difficulty expert --out chart.json
```

## 출력 형식

JSON은 다음 구조입니다.

```json
{
  "metadata": {
    "bpm": 128.0,
    "beat_offset": 0.123,
    "keys": 4,
    "difficulty": "hard"
  },
  "notes": [
    {
      "time": 1.234,
      "lane": 0,
      "kind": "tap",
      "duration": 0.0,
      "beat": 2.0,
      "weight": 0.83
    }
  ]
}
```

`lane`은 0부터 시작합니다. `kind`는 `tap` 또는 `hold`입니다.

## 튜닝 팁

- 노트가 너무 많으면 `--difficulty normal` 또는 `--difficulty easy`를 사용하세요.
- 같은 곡에서 다른 느낌의 패턴을 원하면 `--seed` 값을 바꾸세요.
- 엔진을 게임 포맷에 맞추려면 JSON을 변환하는 어댑터를 추가하면 됩니다.
