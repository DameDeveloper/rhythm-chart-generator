@echo off
echo === Rhythm Chart Generator ===
echo.

where yt-dlp >nul 2>&1 || (echo [!] yt-dlp가 필요합니다: pip install yt-dlp && exit /b 1)
where ffmpeg >nul 2>&1 || (echo [!] ffmpeg가 필요합니다: https://ffmpeg.org/download.html && exit /b 1)

echo [1/2] Backend 시작 중... (http://localhost:8000)
start "backend" cmd /c "cd /d %~dp0backend && pip install -r requirements.txt -q && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Frontend 시작 중... (http://localhost:5173)
start "frontend" cmd /c "cd /d %~dp0frontend && npm install --silent && npm run dev"

echo.
echo 브라우저에서 http://localhost:4000 을 열어주세요.
echo 종료하려면 열린 두 터미널 창을 닫으세요.
pause
