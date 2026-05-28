# NJU Rule RAG 零基础部署全流程指南

> 适用人群：有代码、没有部署经验的同学
> 预计耗时：2～3 小时（含等待时间）
> 最终效果：QQ 群机器人 24 小时在线，月费约 55 元

---

## 目录

1. [准备工作：注册账号、申请 API Key](#一准备工作)
2. [购买云服务器](#二购买云服务器)
3. [连接服务器](#三连接服务器)
4. [服务器初始化](#四服务器初始化)
5. [上传代码](#五上传代码)
6. [配置环境变量](#六配置环境变量)
7. [构建索引（首次必做）](#七构建索引首次必做)
8. [启动服务](#八启动服务)
9. [配置 Nginx 对外暴露接口](#九配置-nginx)
10. [NapCat 登录 QQ Bot](#十napcat-登录-qq-bot)
11. [验证上线](#十一验证上线)
12. [日常维护](#十二日常维护)

---

## 一、准备工作

### 1.1 注册腾讯云账号

1. 打开 https://cloud.tencent.com
2. 右上角「注册」，用手机号注册
3. 完成实名认证（需身份证，约 5 分钟）

### 1.2 申请阿里云百炼 API Key（用于 Qwen3-8B）

1. 打开 https://bailian.console.aliyun.com
2. 用支付宝扫码登录（无需单独注册）
3. 左侧菜单 → 「API-KEY」→「创建 API-KEY」
4. 复制并保存这个 Key（格式类似 `sk-xxxxxxxxxxxxxxxx`）
5. 新用户有免费额度，学生用量基本不花钱

---

## 二、购买云服务器

### 推荐产品

**腾讯云轻量应用服务器**（适合新手，操作最简单）

### 购买步骤

1. 登录腾讯云，进入「轻量应用服务器」产品页
2. 点击「立即选购」，按如下配置选择：

| 选项 | 推荐值 |
|------|--------|
| 地域 | 上海（或离南京最近的） |
| 镜像 | Ubuntu 22.04 LTS |
| 套餐 | **2 核 4 GB 内存 / 60 GB 硬盘 / 4 Mbps 带宽** |
| 购买时长 | 先买 1 个月试试，稳定后再续费 |

3. 结算付款（约 50 元/月，活动期间更便宜）
4. 购买完成后在「控制台 → 轻量应用服务器」找到你的机器，记下**公网 IP**

### 开放防火墙端口

在控制台找到「防火墙」，添加以下入站规则：

| 端口 | 协议 | 用途 |
|------|------|------|
| 22 | TCP | SSH 登录 |
| 80 | TCP | HTTP 访问 |

> 其他端口保持关闭，安全起见不要开放 8000、6099 等内部端口。

---

## 三、连接服务器

### 准备 SSH 工具

- **Windows**：下载安装 [MobaXterm](https://mobaxterm.mobatek.net/download.html)（免费，图形界面，推荐新手）
- **Mac / Linux**：直接用系统自带的「终端」

### 获取登录密码

1. 控制台 → 轻量应用服务器 → 找到你的机器
2. 点击「重置密码」，设置一个 root 密码并记住它

### 连接

```bash
# 替换为你实际的公网 IP
ssh root@你的公网IP
```

输入密码后看到命令行提示符，说明连接成功。

---

## 四、服务器初始化

**以下命令在服务器上逐条执行：**

```bash
# 1. 更新系统（需要等 1～2 分钟）
apt update && apt upgrade -y

# 2. 安装 Docker
curl -fsSL https://get.docker.com | sh

# 3. 安装 Nginx
apt install nginx -y

# 4. 验证 Docker 安装成功
docker --version
# 应输出：Docker version 26.x.x，表示安装成功

# 5. 启动 Nginx
systemctl start nginx
systemctl enable nginx
```

---

## 五、上传代码

### 方式 A：从 GitHub 克隆（推荐）

```bash
# 在服务器上执行
cd /root
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
```

### 方式 B：从本地上传（如果代码在自己电脑上）

在**本地电脑**的终端执行：

```bash
# 将整个项目文件夹上传到服务器（替换路径和 IP）
scp -r /本地/项目路径/nju-rule-rag root@你的公网IP:/root/
```

MobaXterm 用户可以直接拖拽文件夹上传。

---

## 六、配置环境变量

```bash
# 在服务器项目目录下执行
cd /root/nju-rule-rag
cp .env.example .env
nano .env
```

用键盘编辑以下三行（其余保持不变）：

```
LLM_API_KEY=你的阿里云百炼API_KEY
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-8b
```

保存：按 `Ctrl+O`，回车确认，再按 `Ctrl+X` 退出。

---

## 七、构建索引（首次必做）

索引是 RAG 系统的核心，需要在服务器上构建一次。

```bash
cd /root/nju-rule-rag

# 安装 Python 依赖（需要等 3～5 分钟）
pip install -r requirements.txt

# 首次运行会下载中文 embedding 模型（约 400MB，需保持网络畅通）
python scripts/build_index.py
```

> 如果本地已经有构建好的 `data/index/` 目录，可以直接用 scp 上传，跳过此步骤：
>
> ```bash
> scp -r /本地/nju-rule-rag/data root@你的公网IP:/root/nju-rule-rag/
> ```

---

## 八、启动服务

### 8.1 创建 docker-compose.yml

如果项目里没有现成的，在 `/root/nju-rule-rag/` 下创建：

```bash
nano docker-compose.yml
```

粘贴以下内容：

```yaml
services:
  rag-api:
    build: .
    restart: always
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1

  napcat:
    image: mlikiowa/napcat-docker:latest
    restart: always
    ports:
      - "127.0.0.1:6099:6099"
    volumes:
      - ./napcat/config:/app/napcat/config
      - ./napcat/qq:/root/.config/QQ
    environment:
      - NAPCAT_GID=0
      - NAPCAT_UID=0
```

保存退出（`Ctrl+O` → 回车 → `Ctrl+X`）。

### 8.2 启动所有服务

```bash
cd /root/nju-rule-rag
docker compose up -d --build
```

首次启动会构建镜像，需要 5～10 分钟。

### 8.3 检查是否启动成功

```bash
docker compose ps
```

两个服务的 STATUS 都显示 `Up` 即为成功。

---

## 九、配置 Nginx

```bash
nano /etc/nginx/sites-available/rag
```

粘贴：

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }
}
```

保存后执行：

```bash
# 启用配置
ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/rag

# 删除默认配置（避免冲突）
rm -f /etc/nginx/sites-enabled/default

# 测试配置语法
nginx -t

# 重载 Nginx
systemctl reload nginx
```

---

## 十、NapCat 登录 QQ Bot

### 10.1 用 SSH 端口转发在本地访问 NapCat 管理页

**在本地电脑的终端执行**（不是服务器上）：

```bash
ssh -L 6099:127.0.0.1:6099 root@你的公网IP
```

保持这个窗口开着，然后在本地浏览器打开：

```
http://localhost:6099
```

### 10.2 扫码登录 QQ 小号

1. 打开 NapCat 管理页面
2. 点击「扫码登录」
3. 用手机 QQ 扫码（建议用注册好的 QQ 小号，不要用自己主号）

> **遇到「网络环境不稳定」提示怎么办：**
> 先在 Windows 本地登录同一个 QQ 小号，然后将
> `C:\Users\你的用户名\Documents\Tencent Files\QQ号\` 文件夹
> 上传到服务器 `./napcat/qq/` 目录下，再重启 NapCat：
> ```bash
> docker compose restart napcat
> ```

### 10.3 配置消息回调

登录 NapCat WebUI 后：

1. 左侧「网络配置」
2. 添加「HTTP 服务器」→ 端口填 `3000`
3. 添加「HTTP 客户端」→ 上报地址填 `http://rag-api:8000/qq`
4. 保存配置

---

## 十一、验证上线

### 测试 RAG 接口

在服务器上执行：

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 测试问答
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"缓考怎么申请？"}'
```

从公网测试（在本地电脑浏览器或终端）：

```
http://你的公网IP/health
```

### 测试 QQ Bot

在你的 QQ 群里发送：

```
/问 缓考怎么申请？
```

收到【结论】【依据】【提醒】三段式回复，说明全链路跑通了。

---

## 十二、日常维护

### 查看日志

```bash
# 实时查看 RAG 服务日志
docker compose logs -f rag-api

# 查看 NapCat 日志
docker compose logs -f napcat
```

### 重启服务

```bash
cd /root/nju-rule-rag
docker compose restart
```

### 更新代码

```bash
cd /root/nju-rule-rag
git pull
docker compose up -d --build
```

### 服务器重启后自动恢复

docker-compose.yml 里已经设置了 `restart: always`，服务器重启后 Docker 服务会自动拉起所有容器，无需手动操作。

---

## 常见问题

| 现象 | 原因 | 解决办法 |
|------|------|---------|
| `docker compose up` 报错 | 代码或依赖问题 | `docker compose logs rag-api` 看具体报错 |
| 访问公网 IP 没有响应 | Nginx 没启动或端口没开 | `systemctl status nginx`，检查控制台防火墙 |
| QQ Bot 没有回复 | NapCat 回调地址配错 | 检查是否填写 `http://rag-api:8000/qq` |
| 回答质量差 | embedding 模型没下载完整 | 重新执行 `python scripts/build_index.py` |
| LLM 超时 | API Key 错误或网络波动 | 检查 `.env` 中的 Key，项目已有 3 次自动重试 |

---

## 费用总结

| 项目 | 月费用 |
|------|--------|
| 腾讯云轻量 2C4G | ~50 元 |
| 阿里云百炼 Qwen3-8B API | ~0～5 元（新用户有免费额度） |
| **合计** | **约 50～55 元/月** |
