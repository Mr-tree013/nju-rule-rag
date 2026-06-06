@echo off
REM NJU Rule RAG — Windows startup script
REM Run this from D:\project\nju-rule-rag

echo === NJU Rule RAG — Windows Startup ===

REM Set Ollama environment (must be set before ollama serve)
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_KEEP_ALIVE=24h

REM Activate venv and start server
call .venv\Scripts\activate.bat

echo === Starting uvicorn on http://0.0.0.0:8000 ===
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
