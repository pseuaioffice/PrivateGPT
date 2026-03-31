@echo off
title MyAIAssistant - Starting...
color 0A

echo.
echo  ========================================
echo       MyAIAssistant Desktop Application
echo  ========================================
echo.
echo  Starting Backend Server...

:: Get the directory where this batch file lives
set "APP_DIR=%~dp0"

:: Start backend in background
start "MyAIAssistant Backend" /MIN "%APP_DIR%backend\backend.exe"

:: Wait for backend to start
echo  Waiting for Backend to initialize...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 goto wait_backend
echo  [OK] Backend is running!

:: Start frontend in background
echo  Starting Frontend Server...
start "MyAIAssistant Frontend" /MIN "%APP_DIR%frontend\frontend.exe"

:: Wait a moment for frontend
timeout /t 3 /nobreak >nul
echo  [OK] Frontend is running!

:: Open browser
echo  Opening browser...
start "" http://127.0.0.1:5000

title MyAIAssistant - Running
color 0A
echo.
echo  ========================================
echo   MyAIAssistant is running!
echo   App:     http://127.0.0.1:5000
echo   API:     http://127.0.0.1:8000
echo  ========================================
echo.
echo   Keep this window open.
echo   Press any key to SHUT DOWN MyAIAssistant.
echo.
pause >nul

:: Cleanup
echo  Shutting down...
:: Kill by executable name instead of Window Title. 
:: PyInstaller console apps do not reliably match the WINDOWTITLE filter.
taskkill /IM "backend.exe" /F >nul 2>&1
taskkill /IM "frontend.exe" /F >nul 2>&1
echo  Done. Goodbye!
timeout /t 2 /nobreak >nul
