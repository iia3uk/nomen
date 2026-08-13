@echo off
REM Double-click = resume previous hunt if nomen_results\checkpoint.json exists.
REM Fresh hunt: run.bat --fresh
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if not exist "%PY%" set "PY=python"
    echo Creating virtualenv...
    "%PY%" -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Install Python 3.12 and retry.
        goto :finish
    )
)

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

set PYTHONIOENCODING=utf-8
chcp 65001 >nul

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 goto :finish

".venv\Scripts\python.exe" -m nomen %*
set "ERR=%ERRORLEVEL%"

:finish
if not defined ERR set "ERR=%ERRORLEVEL%"
echo %CMDCMDLINE% | find /i "/c" >nul
if not errorlevel 1 (
    echo.
    pause
)
exit /b %ERR%
