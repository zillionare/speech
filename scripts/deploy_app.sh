#!/usr/bin/env bash
#
# 部署完整的 speech-studio（Python 服务 + web 前端）到 pro。
#
# 用法:
#   scripts/deploy_app.sh              # 构建前端 + 同步源码 + 生成配置 + 重启服务
#   scripts/deploy_app.sh --skip-web   # 跳过前端构建/同步（前端未改动时更快）
#   scripts/deploy_app.sh --no-restart # 只同步与生成配置，不重启服务
#
# 环境变量:
#   PRO_HOST  部署机地址，默认 quantide@192.168.0.102
#   PRO_DIR   部署路径，默认 ~/apps/speech-studio
#   PRO_PASS  部署机密码；设置后走 sshpass，不设置则走 ssh 密钥/交互式登录
#   PRO_CERT_HOST  自签证书的 IP SAN，默认 192.168.0.102
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRO_HOST="${PRO_HOST:-quantide@192.168.0.102}"
PRO_DIR="${PRO_DIR:-~/apps/speech-studio}"
PRO_CERT_HOST="${PRO_CERT_HOST:-192.168.0.102}"

SKIP_WEB=0
NO_RESTART=0
for arg in "$@"; do
    case "$arg" in
        --skip-web) SKIP_WEB=1 ;;
        --no-restart) NO_RESTART=1 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [[ -n "${PRO_PASS:-}" ]]; then
    command -v sshpass >/dev/null || { echo "未找到 sshpass，请改用 ssh 密钥或去掉 PRO_PASS" >&2; exit 1; }
    RSYNC_SHELL="sshpass -p $PRO_PASS ssh $SSH_OPTS"
    ssh_run() { sshpass -p "$PRO_PASS" ssh $SSH_OPTS "$@"; }
else
    RSYNC_SHELL="ssh $SSH_OPTS"
    ssh_run() { ssh $SSH_OPTS "$@"; }
fi

if [[ "$SKIP_WEB" == "0" ]]; then
    echo "==> 构建前端 (web/)"
    cd "$ROOT/web"
    pnpm install --frozen-lockfile
    pnpm build

    echo "==> 同步 web/dist -> $PRO_HOST:$PRO_DIR/web/dist"
    ssh_run "$PRO_HOST" "mkdir -p $PRO_DIR/web/dist"
    rsync -az --delete -e "$RSYNC_SHELL" "$ROOT/web/dist/" "$PRO_HOST:$PRO_DIR/web/dist/"
fi

echo "==> 同步后端源码 -> $PRO_HOST:$PRO_DIR"
ssh_run "$PRO_HOST" "mkdir -p $PRO_DIR"
rsync -az \
    --exclude='.venv' --exclude='web' --exclude='outputs' --exclude='.git' \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
    --exclude='.certs' --exclude='*.gguf' --exclude='.speech-secrets.yaml' \
    --exclude='.DS_Store' \
    -e "$RSYNC_SHELL" "$ROOT/" "$PRO_HOST:$PRO_DIR/"

echo "==> 生成 pro 版配置 / secrets / 自签证书"
ssh_run "$PRO_HOST" "
    set -euo pipefail
    cd $PRO_DIR
    mkdir -p outputs .certs
    if [ ! -f .certs/dev-key.pem ]; then
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout .certs/dev-key.pem -out .certs/dev-cert.pem \
            -days 365 -subj '/CN=speech-studio' \
            -addext 'subjectAltName=IP:$PRO_CERT_HOST,IP:127.0.0.1,DNS:localhost' >/dev/null 2>&1
        echo '  已生成自签证书'
    fi
    printf 'yovoice_api_token: %s\n' \"\$(cat ~/.yovoice-token)\" > .speech-secrets.yaml
    chmod 600 .speech-secrets.yaml
    .venv/bin/python scripts/pro_config.py --config config.yaml --base-dir \"\$PWD\"
"

if [[ "$NO_RESTART" == "1" ]]; then
    echo "==> 已跳过重启（--no-restart）"
    exit 0
fi

echo "==> 重启服务"
ssh_run "$PRO_HOST" "
    cd $PRO_DIR
    pkill -f 'tts_service.server_runner' 2>/dev/null || true
    sleep 1
    nohup .venv/bin/python -m tts_service.server_runner --config config.yaml > /tmp/speech.log 2>&1 &
    sleep 5
    tail -3 /tmp/speech.log
    curl -sk -m 10 https://127.0.0.1:8443/health && echo
"
echo "==> 完成，测试地址 https://$PRO_CERT_HOST:8443/"
