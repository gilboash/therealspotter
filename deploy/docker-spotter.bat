@echo off
rem ===================================================================
rem  fpv-spotter (watch-only page): build the image, (re)start the container.
rem
rem  Lives in the app's folder on the host (deploy-optiplex.sh puts it there):
rem    build\   what the image is built from (Dockerfile.view, the code in app\)
rem    data\    the copy of the analysed data, mounted at /data: never in the image
rem    .env     SPOTTER_KEY=<the spotter code>, never in the image or the repository
rem  It only touches things named fpv-spotter. --memory: the box is shared, so the page can't take more than 1 GB
rem  (it levels off at ~500 MB; over 1 GB Docker restarts it).
rem ===================================================================
setlocal
set "APP=%~dp0"
if "%APP:~-1%"=="\" set "APP=%APP:~0,-1%"
set "DATA=%APP%\data"
if "%PORT%"=="" set PORT=5190
if not exist "%DATA%" mkdir "%DATA%"
if not exist "%APP%\.env" (echo SPOTTER_KEY=>"%APP%\.env")

echo [1/4] building the image (the previous one is kept as fpv-spotter:prev, to roll back to)
cd /d "%APP%\build"
docker tag fpv-spotter:latest fpv-spotter:prev >nul 2>&1
docker build -t fpv-spotter:latest -f Dockerfile.view . || goto :failed

echo [2/4] replacing the container
docker rm -f fpv-spotter >nul 2>&1

echo [3/4] starting on port %PORT%
docker run -d ^
  --name fpv-spotter ^
  --restart unless-stopped ^
  --memory 1g --memory-swap 1g ^
  -p 0.0.0.0:%PORT%:5190 ^
  -v "%DATA%:/data" ^
  --env-file "%APP%\.env" ^
  fpv-spotter:latest || goto :failed

echo [4/4] waiting for it to answer
powershell -NoProfile -Command "Start-Sleep -Seconds 8"
curl.exe -s -m 20 -o NUL -w "      page HTTP %%{http_code}\n" http://127.0.0.1:%PORT%/
curl.exe -s -m 60 -o NUL -w "      tracks HTTP %%{http_code}, %%{size_download} bytes\n" http://127.0.0.1:%PORT%/api/tracks
docker ps --filter name=fpv-spotter --format "      {{.Names}}  {{.Status}}  {{.Ports}}"
exit /b 0

:failed
echo.
echo FAILED - see the output above. The data in %DATA% is untouched (a mount).
echo   docker logs fpv-spotter
echo   roll back: docker tag fpv-spotter:prev fpv-spotter:latest, then run this script's docker run line
exit /b 1
