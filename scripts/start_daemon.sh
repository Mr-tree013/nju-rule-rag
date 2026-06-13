#!/bin/bash
# NJU Rule RAG — 后台服务一键启动
# 使用：bash scripts/start_daemon.sh

SESSION="nju-rag"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 设置 pip 镜像 ──────────────────────────────────
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF
echo "[0/3] pip 镜像已设为阿里云"

# ── 确保 Ollama 在运行 ──────────────────────────────
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[1/3] 启动 Ollama..."
    ollama serve &>/tmp/ollama.log &
    sleep 3
else
    echo "[1/3] Ollama 已在运行"
fi

# ── 启动服务器 ─────────────────────────────────────
USE_TMUX=false
command -v tmux &>/dev/null && USE_TMUX=true

if $USE_TMUX; then
    # tmux 模式：关终端不中断，可随时 attach 查看
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[2/3] tmux 会话 $SESSION 已存在"
    else
        echo "[2/3] 创建 tmux 会话并启动服务器..."
        cd "$ROOT"
        tmux new-session -d -s "$SESSION" -n server
        tmux send-keys -t "$SESSION" "source .venv/bin/activate && ./scripts/start_server.sh" Enter
        sleep 2
        tmux new-window -t "$SESSION" -n monitor
        tmux send-keys -t "$SESSION:monitor" \
          "watch -n 5 'curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null'" Enter
    fi
else
    # nohup 模式：不需要 tmux，也能关终端
    echo "[2/3] tmux 未安装，使用 nohup 模式启动..."
    cd "$ROOT"
    source .venv/bin/activate
    nohup ./scripts/start_server.sh &>/tmp/server.log &
    disown
fi

echo "[3/3] 启动完成"
echo ""
echo "═══════════════════════════════════════════"
echo "  服务已在后台运行，你可以关闭屏幕了"
echo ""
echo "  下次打开终端检查状态："
echo "    curl http://localhost:8000/health"
if $USE_TMUX; then
    echo ""
    echo "  tmux 命令："
    echo "    tmux attach -t $SESSION         # 进入会话"
    echo "    tmux ls                          # 列出会话"
    echo "    tmux kill-session -t $SESSION    # 关闭服务"
fi
echo "═══════════════════════════════════════════"
