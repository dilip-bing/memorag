@echo off
echo Stopping existing Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Starting RAG API server...
cd /d "%~dp0"
start /b python api.py

echo Backend restarted!
echo Check logs at: logs\rag.log
