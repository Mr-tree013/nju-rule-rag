#!/bin/bash
# NJU Rule RAG — 后台服务一键启动（tmux 会话模式）
# 使用：bash scripts/start_daemon.sh
# 关闭屏幕后服务继续运行，下次用 tmux attach 查看状态

SESSION="nju-rag"

# 确保 Ollama 在运行
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[1/3] 启动 Ollama..."
    ollama serve &>/tmp/ollama.log &
    sleep 3
else
    echo "[1/3] Ollama 已在运行"
fi

# 在 tmux 会话中启动服务器
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[2/3] tmux 会话 $SESSION 已存在，连接到已有会话"
else
    echo "[2/3] 创建 tmux 会话并启动服务器..."
    cd "$(dirname "$0")/.."
    tmux new-session -d -s "$SESSION" -n server
    tmux send-keys -t "$SESSION" "source .venv/bin/activate && ./scripts/start_server.sh" Enter
    sleep 2
    tmux new-window -t "$SESSION" -n monitor
    tmux send-keys -t "$SESSION:monitor" "watch -n 5 'curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null'" Enter
fi

echo "[3/3] 启动完成"
echo ""
echo "═══════════════════════════════════════════"
echo "  服务已在后台运行，你可以关闭屏幕了"
echo ""
echo "  下次打开终端时，用以下命令查看状态："
echo "    tmux attach -t $SESSION         # 进入会话"
echo "    tmux ls                          # 列出会话"
echo "    curl http://localhost:8000/health # 检查服务"
echo ""
echo "  关闭服务："
echo "    tmux kill-session -t $SESSION"
echo "═══════════════════════════════════════════"
