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

"C:\Users\andre\venv_bolas\Scripts\python.exe" "%~dp0MasterControl.py" %*

echo.
echo  ----------------------------------------
echo   Sistema terminado
echo  ----------------------------------------
taskkill /f /im python.exe /t >nul 2>&1
pause
