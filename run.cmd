@echo off
setlocal

if "%~1"=="" goto help
if "%~1"=="help" goto help
if "%~1"=="up" goto up
if "%~1"=="build" goto build_sample
if "%~1"=="build-sample" goto build_sample
if "%~1"=="migrate" goto migrate
if "%~1"=="test" goto test
if "%~1"=="stop" goto stop
if "%~1"=="down" goto down
if "%~1"=="logs" goto logs
if "%~1"=="ps" goto ps
if "%~1"=="restart" goto restart

echo Unknown command: %~1
goto help

:up
echo Starting Song Search stack in Docker...
docker compose up -d --build
echo.
echo All services started!
echo - Web App:        http://localhost:8000
echo - Milvus Health:  http://localhost:9091/healthz
echo - MinIO Console:  http://localhost:9001 (minioadmin / minioadmin)
exit /b 0

:migrate
echo Migrating FAISS sample into Milvus...
docker compose run --rm migrate
exit /b %ERRORLEVEL%

:build_sample
echo Building 1,000-song sample index into Milvus...
docker compose run --rm build-sample
exit /b %ERRORLEVEL%

:test
echo Running tests in Docker...
docker compose run --rm test
exit /b %ERRORLEVEL%

:stop
echo Stopping containers (preserving data)...
docker compose stop
exit /b 0

:down
echo Stopping and removing containers (preserving data volumes)...
docker compose down
exit /b 0

:logs
docker compose logs -f web
exit /b 0

:ps
docker compose ps
exit /b 0

:restart
docker compose restart
exit /b 0

:help
echo Song Search - Docker Helper
echo.
echo Usage: run.cmd [command]
echo.
echo Commands:
echo   up           Build and start all services (Milvus + MinIO + etcd + Web App)
echo   migrate      Migrate artifacts/sample into Milvus (songs_sample)
echo   build        Build 1,000-song sample index from CSV into Milvus
echo   test         Run test suite inside the Docker container
echo   logs         Follow FastAPI web application logs
echo   ps           Check status of all containers
echo   stop         Stop running containers (keeps your data)
echo   restart      Restart running containers
echo   down         Take down containers (keeps your data volumes)
exit /b 0
