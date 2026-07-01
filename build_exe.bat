@echo off
setlocal
echo === Rhythm Chart Generator - EXE 빌드 ===
echo.

cd /d "%~dp0"

echo [1/3] 프론트엔드 빌드...
pushd frontend
call npm install --silent
call npx vite build || (echo [!] 프론트엔드 빌드 실패 & popd & exit /b 1)
popd

echo [2/3] 빌드 도구 설치 (PyInstaller + 백엔드 의존성)...
python -m pip install -q pyinstaller -r backend\requirements.txt || (echo [!] 의존성 설치 실패 & exit /b 1)

echo [3/3] 단일 실행파일 패키징...
pushd backend
python -m PyInstaller rhythm_chart_generator.spec --noconfirm --distpath "..\dist_exe" --workpath "..\build_exe" || (echo [!] 패키징 실패 & popd & exit /b 1)
popd

echo.
echo === 완료 ===
echo 실행파일: dist_exe\RhythmChartGenerator.exe
echo 더블클릭하면 서버가 켜지고 브라우저가 자동으로 열립니다.
pause
