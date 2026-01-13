#!/bin/bash
# 诊断脚本：检查环境变量

echo "========================================"
echo "环境变量诊断"
echo "========================================"
echo "START_ALTITUDE_OFFSET = ${START_ALTITUDE_OFFSET:-未设置}"
echo "GOAL_ALTITUDE = ${GOAL_ALTITUDE:-未设置}"
echo "HEIGHT_IDEAL_MIN = ${HEIGHT_IDEAL_MIN:-未设置}"
echo "HEIGHT_IDEAL_MAX = ${HEIGHT_IDEAL_MAX:-未设置}"
echo ""
echo "固定位置文件内容:"
cat saved_positions/5.json 2>/dev/null || echo "文件不存在"
echo ""
echo "========================================"

