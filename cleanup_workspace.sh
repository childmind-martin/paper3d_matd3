#!/bin/bash

echo "=== 清理工作空间 ==="
echo ""

cd /home/tang/Desktop

# 1. 清理未跟踪的 md 文件（保留 Git 中已跟踪的重要文档）
echo "1. 检查未跟踪的 MD 文件..."
KEEP_MD=(
    "MODULE_STRUCTURE.md"
    "PER_Paper_Description.md"
    "QUADROTOR_DYNAMICS_README.md"
    "README_MODULES.md"
    "terrain_visualizer_README.md"
)

UNTRACKED_MD=$(find . -maxdepth 1 -name "*.md" -type f | while read file; do
    basename_file=$(basename "$file")
    if ! git ls-files --error-unmatch "$basename_file" >/dev/null 2>&1; then
        echo "$basename_file"
    fi
done)

if [ -n "$UNTRACKED_MD" ]; then
    echo "发现未跟踪的 MD 文件："
    echo "$UNTRACKED_MD" | while read file; do
        echo "  ✗ $file"
    done
    echo ""
    echo "删除这些文件..."
    echo "$UNTRACKED_MD" | while read file; do
        rm -f "$file" && echo "  已删除: $file"
    done
else
    echo "  没有未跟踪的 MD 文件"
fi

echo ""

# 2. 清理 logs 中空或几乎空的文件夹（小于 100KB）
echo "2. 检查 logs 目录中的空文件夹..."
EMPTY_LOGS=0
TOTAL_SIZE=0

for dir in logs/*/; do
    if [ -d "$dir" ]; then
        size=$(du -sk "$dir" 2>/dev/null | cut -f1)
        if [ "$size" -lt 100 ]; then
            EMPTY_LOGS=$((EMPTY_LOGS + 1))
            TOTAL_SIZE=$((TOTAL_SIZE + size))
        fi
    fi
done

echo "  发现 $EMPTY_LOGS 个空或几乎空的文件夹（< 100KB）"
echo "  总大小: ${TOTAL_SIZE}KB"

if [ "$EMPTY_LOGS" -gt 0 ]; then
    echo ""
    read -p "确认删除这些空文件夹吗？(y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DELETED=0
        for dir in logs/*/; do
            if [ -d "$dir" ]; then
                size=$(du -sk "$dir" 2>/dev/null | cut -f1)
                if [ "$size" -lt 100 ]; then
                    rm -rf "$dir"
                    DELETED=$((DELETED + 1))
                    if [ $((DELETED % 50)) -eq 0 ]; then
                        echo "  已删除 $DELETED 个文件夹..."
                    fi
                fi
            fi
        done
        echo "  完成！共删除 $DELETED 个空文件夹"
    else
        echo "  取消删除"
    fi
else
    echo "  没有需要清理的空文件夹"
fi

echo ""
echo "=== 清理完成 ==="
