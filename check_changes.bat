@echo off
echo Checking git status for local-rag...
echo.
cd /d "%~dp0"
git status
echo.
echo ===================================
echo Press any key to see detailed diff...
pause
git diff
