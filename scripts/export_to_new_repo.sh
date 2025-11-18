#!/usr/bin/env bash
set -euo pipefail

# 导出当前 footrl 目录为一个独立仓库并推送到远程
# 用法:
#   bash scripts/export_to_new_repo.sh /absolute/path/to/output sim_footrl.git
# 可选 --dry-run 仅构建目录不执行 git push
# 远程地址示例: git@github.com:kongbai666ciallo/sim_footrl.git 或 https://github.com/kongbai666ciallo/sim_footrl.git

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <输出目录> <remote_url> [--dry-run]" >&2
  exit 1
fi

OUTPUT_DIR="$1"
REMOTE_URL="$2"
DRY_RUN="false"
if [[ "${3:-}" == "--dry-run" ]]; then
  DRY_RUN="true"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOOTRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -e "$OUTPUT_DIR/.git" ]]; then
  echo "目标目录已存在 git 仓库: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rsync -av --exclude ".git" --exclude "__pycache__" --exclude ".pytest_cache" \
  "$FOOTRL_ROOT/" "$OUTPUT_DIR/"

cd "$OUTPUT_DIR"

echo "初始化 git 仓库..."
git init -b main

echo "写入 .gitignore (如果不存在)"
if [[ ! -f .gitignore ]]; then
  cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
*.swp
.env
.DS_Store
.vscode/
logs/
outputs/
EOF
fi

echo "添加所有文件并提交"
git add .
git commit -m "Initial import of footrl subpackage"

echo "设置远程: $REMOTE_URL"
git remote add origin "$REMOTE_URL"

echo "当前提交: $(git rev-parse --short HEAD)"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry-run 模式：不执行 push。可在确认后运行: git push -u origin main"
  exit 0
fi

echo "推送到远程..."
git push -u origin main

echo "完成。打开 GitHub 检查仓库: $REMOTE_URL"
