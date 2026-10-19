#!/usr/bin/env bash
#
# 构建 vendor 的 yovoice 前端（web/），并把产物同步到部署机 pro。
#
# 用法:
#   scripts/deploy_web.sh                 # 构建 + 同步
#   scripts/deploy_web.sh --build-only    # 只构建，不同步
#
# 环境变量:
#   PRO_HOST  部署机地址，默认 quantide@192.168.0.102
#   PRO_DIR   部署路径，默认 ~/apps/speech-studio
#   PRO_PASS  部署机密码；设置后走 sshpass，不设置则走 ssh 密钥/交互式登录
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRO_HOST="${PRO_HOST:-quantide@192.168.0.102}"
PRO_DIR="${PRO_DIR:-~/apps/speech-studio}"

BUILD_ONLY=0
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=1

echo "==> 构建前端 (web/)"
cd "$ROOT/web"
pnpm install --frozen-lockfile
pnpm build

if [[ "$BUILD_ONLY" == "1" ]]; then
    echo "==> 仅构建，产物在 $ROOT/web/dist"
    exit 0
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [[ -n "${PRO_PASS:-}" ]]; then
    command -v sshpass >/dev/null || { echo "未找到 sshpass，请改用 ssh 密钥或去掉 PRO_PASS" >&2; exit 1; }
    RSYNC_SHELL="sshpass -p $PRO_PASS ssh $SSH_OPTS"
else
    RSYNC_SHELL="ssh $SSH_OPTS"
fi

echo "==> 同步 web/dist -> $PRO_HOST:$PRO_DIR/web/dist"
# 确保目标目录存在
if [[ -n "${PRO_PASS:-}" ]]; then
    sshpass -p "$PRO_PASS" ssh $SSH_OPTS "$PRO_HOST" "mkdir -p $PRO_DIR/web/dist"
else
    ssh $SSH_OPTS "$PRO_HOST" "mkdir -p $PRO_DIR/web/dist"
fi

rsync -az --delete -e "$RSYNC_SHELL" "$ROOT/web/dist/" "$PRO_HOST:$PRO_DIR/web/dist/"
echo "==> 完成"
