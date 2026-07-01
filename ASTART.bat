@echo off
REM =====================================================
REM   SISTEMA UFSC/FEUP - Bolas v3 (arranque normal)
REM   Apenas caracteres ASCII para evitar problemas de
REM   encoding no cmd.exe.
REM =====================================================

chcp 65001 > nul
title SISTEMA UFSC/FEUP - Bolas v3

echo.
echo  [ARRANQUE] A verificar processos anteriores...
taskkill /f /im python.exe /t >nul 2>&1
timeout /t 2 /nobreak >nul

echo  [ARRANQUE] A iniciar MasterControl...
echo.

set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo  [ERRO] Ambiente virtual nao encontrado em .venv
    echo  Executa primeiro: py -3.12 -m venv .venv
    echo  Depois: .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0_APP\master_control.py" %*

echo.
echo  ----------------------------------------
echo   Sistema terminado
echo  ----------------------------------------
taskkill /f /im python.exe /t >nul 2>&1
pause





