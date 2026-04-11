cd C:\Users\dilip\OneDrive\Desktop\local-rag

# Check what changed
git status

# Add all changes
git add config.yaml rag/query/engine.py api.py STREAMING.md restart.bat

# Commit
git commit -m "feat: streaming backend + config optimization

- Config: vector-only retrieval, context_window=24576, top_k=4
- Engine: Add streaming support + response_mode=compact
- API: Add /query/stream SSE endpoint
- Show Thinking/Fast thinking status labels

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Push to GitHub
git push origin main@echo off
echo Stopping existing Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Starting RAG API server...
cd /d "%~dp0"
start /b python api.py

echo Backend restarted!
echo Check logs at: logs\rag.log
