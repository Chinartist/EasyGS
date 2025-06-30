#!/bin/bash

# 设置提交信息
if [ -z "$1" ]; then
  commit_message="auto commit: $(date '+%Y-%m-%d %H:%M:%S')"
else
  commit_message="$1   Date: $(date '+%Y-%m-%d %H:%M:%S')"
fi

# 切换到项目目录
cd $(dirname "$0") || exit

# 检查 Git 状态
if [ -n "$(git status --porcelain)" ]; then
  echo "Changes detected. Preparing to commit..."

  # 添加所有更改
  git add .

  # 提交更改
  git commit -m "$commit_message"

  # 推送到远程仓库
  git push origin main  # 如果你的主分支不是 main，请替换为 master 或其他分支名

  echo "Changes have been committed and pushed."
else
  echo "No changes detected."
fi