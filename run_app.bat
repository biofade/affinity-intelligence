@echo off
setlocal EnableExtensions
title Affinity Intelligence - Local Demo
chcp 65001 >nul 2>&1

REM ============================================================================
REM  Affinity Intelligence - one-click launcher (Streamlit demo)
REM  Just double-click this file. No command-line knowledge required.
REM ============================================================================

cd /d "F:\DSH_Galatea\Affinity_Intelligence\GitHub_Web_Upload"
if errorlevel 1 goto :badpath

REM --- help flag (non-interactive check, does not start the app) -------------
if /i "%~1"=="--help" goto :showhelp
if /i "%~1"=="-h" goto :showhelp

echo ============================================================
echo   Affinity Intelligence - protein-protein affinity ranking
echo ============================================================
echo   Working directory: %CD%
echo.

REM --- locate a usable Python interpreter (py launcher or python) -----------
set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py"
if not defined PY_CMD (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python was not found on this computer.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    echo         IMPORTANT: tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo   Python launcher: %PY_CMD%
%PY_CMD% --version
echo.

REM --- check that Streamlit is installed --------------------------------
%PY_CMD% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed in this Python environment.
    echo.
    echo         Fix it by running this command once:
    echo             %PY_CMD% -m pip install -r requirements.txt
    echo.
    choice /C YN /M "         Install dependencies now"
    if errorlevel 2 goto :nodeps
    echo.
    echo   Installing dependencies (this may take a few minutes)...
    %PY_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed. See messages above.
        pause
        exit /b 1
    )
)
goto :launch

:nodeps
echo.
echo   Cancelled. Please install the dependencies and run this file again.
pause
exit /b 1

:launch
echo   Starting the web app. Your browser will open at http://localhost:8501
echo.
echo   NOTE 1: keep this window OPEN while using the app.
echo   NOTE 2: the FIRST prediction takes about 3 seconds; later ones are faster.
echo   NOTE 3: press Ctrl+C in this window to stop the app.
echo.
echo ------------------------------------------------------------
%PY_CMD% -m streamlit run app.py

echo.
echo   The app has stopped.
pause
exit /b 0

:badpath
echo.
echo [ERROR] Project folder not found:
echo         F:\DSH_Galatea\Affinity_Intelligence\GitHub_Web_Upload
echo.
echo         If you moved the project, edit the "cd /d" line at the top of
echo         this file and point it to your new folder.
echo.
pause
exit /b 1

:showhelp
echo Affinity Intelligence - one-click launcher
echo.
echo   run_app.bat            start the Streamlit demo app (default)
echo   run_app.bat --help     show this help and exit
echo.
echo   Working directory: %CD%
echo   Launch command   : ^<python^> -m streamlit run app.py
exit /b 0
