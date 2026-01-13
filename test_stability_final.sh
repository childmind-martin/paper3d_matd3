#!/bin/bash
# 最终稳定性测试：分三阶段验证 XLA 和 AMP 的可行性
# 包含完整的日志记录和错误分析

set -e  # 遇到错误时停止
cd /home/tang/Desktop

# 创建测试结果目录
TEST_DIR="./stability_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

# 日志文件
SUMMARY_LOG="${TEST_DIR}/test_summary.log"
FULL_LOG="${TEST_DIR}/full_test.log"

echo "=========================================="
echo "稳定性测试方案"
echo "=========================================="
echo "测试结果将保存到: $TEST_DIR"
echo ""
echo "测试1：无 XLA + BF16 + TF32（最稳定配置）"
echo "  - 预期速度：70-80s/回合"
echo "  - 预期稳定性：⭐⭐⭐⭐⭐"
echo ""
echo "测试2：XLA + FP32（验证 XLA 基础稳定性）"
echo "  - 预期速度：60-70s/回合"
echo "  - 预期稳定性：⭐⭐⭐"
echo ""
echo "测试3：XLA + BF16（最快配置，需验证）"
echo "  - 预期速度：40-60s/回合"
echo "  - 预期稳定性：⭐（可能崩溃）"
echo "=========================================="
echo "" | tee "$SUMMARY_LOG"

# 记录系统信息
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "系统信息" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "测试时间: $(date)" | tee -a "$SUMMARY_LOG"
echo "TensorFlow 版本: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo '未知')" | tee -a "$SUMMARY_LOG"
echo "CUDA 版本: $(nvcc --version 2>/dev/null | grep 'release' || echo '未知')" | tee -a "$SUMMARY_LOG"
echo "GPU 信息:" | tee -a "$SUMMARY_LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$SUMMARY_LOG" || echo "无法获取GPU信息" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 函数：运行测试并记录结果
run_test() {
    local test_num=$1
    local test_name=$2
    local use_xla=$3
    local amp_mode=$4
    local expected_speed=$5
    shift 5
    local extra_args="$@"
    
    local test_log="${TEST_DIR}/test${test_num}_${test_name}.log"
    local error_log="${TEST_DIR}/test${test_num}_${test_name}_error.log"
    local timing_log="${TEST_DIR}/test${test_num}_${test_name}_timing.txt"
    
    echo "=========================================" | tee -a "$SUMMARY_LOG"
    echo "测试${test_num}: ${test_name}" | tee -a "$SUMMARY_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG"
    echo "配置: USE_XLA=${use_xla}, AMP_MODE=${amp_mode}" | tee -a "$SUMMARY_LOG"
    echo "额外参数: ${extra_args}" | tee -a "$SUMMARY_LOG"
    echo "预期速度: ${expected_speed}" | tee -a "$SUMMARY_LOG"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_LOG"
    echo "" | tee -a "$SUMMARY_LOG"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行测试
    local exit_code=0
    (
        # 设置环境变量并运行
        export USE_XLA=${use_xla}
        export AMP_MODE=${amp_mode}
        export OPTIMIZER_JIT=0
        export JIT_COMPILE=1
        # 解析并导出额外的环境变量
        if [ -n "$extra_args" ]; then
            eval "export $extra_args"
        fi
        ./run_optimized.sh 5 1024 "test${test_num}_${test_name}" 1 2>&1 | tee "$test_log"
    ) || exit_code=$?
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_LOG"
    echo "总耗时: ${duration}秒" | tee -a "$SUMMARY_LOG"
    echo "退出码: ${exit_code}" | tee -a "$SUMMARY_LOG"
    
    # 提取错误信息
    if [ $exit_code -ne 0 ]; then
        echo "" | tee -a "$SUMMARY_LOG"
        echo "❌ 测试${test_num} 失败" | tee -a "$SUMMARY_LOG"
        echo "" | tee -a "$SUMMARY_LOG"
        echo "错误摘要:" | tee -a "$SUMMARY_LOG"
        
        # 提取关键错误信息
        grep -i "error\|exception\|failed\|cuda_error\|traceback" "$test_log" | tail -20 > "$error_log" 2>/dev/null || echo "无法提取错误信息" > "$error_log"
        cat "$error_log" | tee -a "$SUMMARY_LOG"
        
        echo "" | tee -a "$SUMMARY_LOG"
        echo "完整日志: $test_log" | tee -a "$SUMMARY_LOG"
        echo "错误日志: $error_log" | tee -a "$SUMMARY_LOG"
        echo "" | tee -a "$SUMMARY_LOG"
        
        return 1
    else
        echo "" | tee -a "$SUMMARY_LOG"
        echo "✅ 测试${test_num} 通过" | tee -a "$SUMMARY_LOG"
        
        # 提取性能信息（每回合时间）
        echo "" | tee -a "$SUMMARY_LOG"
        echo "性能统计:" | tee -a "$SUMMARY_LOG"
        grep "Time=" "$test_log" | tail -5 > "$timing_log" 2>/dev/null || echo "无法提取时间信息" > "$timing_log"
        
        # 计算平均时间
        local avg_time=$(grep -oP "Time=\K[0-9.]+" "$test_log" | tail -5 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        echo "  平均每回合耗时: ${avg_time}秒" | tee -a "$SUMMARY_LOG"
        
        # 提取最后5回合的时间
        echo "  最后5回合耗时:" | tee -a "$SUMMARY_LOG"
        cat "$timing_log" | tee -a "$SUMMARY_LOG"
        
        echo "" | tee -a "$SUMMARY_LOG"
        echo "完整日志: $test_log" | tee -a "$SUMMARY_LOG"
        echo "" | tee -a "$SUMMARY_LOG"
        
        return 0
    fi
}

# 测试1：无 XLA + BF16 + TF32
if run_test 1 "no_xla_bf16_tf32" 0 "bf16" "70-80s/回合" \
    "TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 NVIDIA_TF32_OVERRIDE=1"; then
    
    echo "继续测试2..." | tee -a "$SUMMARY_LOG"
    echo "" | tee -a "$SUMMARY_LOG"
    
    # 测试2：XLA + FP32
    if run_test 2 "xla_fp32" 1 "off" "60-70s/回合" \
        "XLA_COMPILE_MODE=parallel"; then
        
        echo "继续测试3..." | tee -a "$SUMMARY_LOG"
        echo "" | tee -a "$SUMMARY_LOG"
        
        # 测试3：XLA + BF16
        if run_test 3 "xla_bf16" 1 "bf16" "40-60s/回合" \
            "XLA_COMPILE_MODE=parallel TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 NVIDIA_TF32_OVERRIDE=1"; then
            
            echo "=========================================" | tee -a "$SUMMARY_LOG"
            echo "🎉 所有测试通过！" | tee -a "$SUMMARY_LOG"
            echo "=========================================" | tee -a "$SUMMARY_LOG"
            echo "推荐配置：XLA + BF16 + TF32 + 并行编译" | tee -a "$SUMMARY_LOG"
            echo "" | tee -a "$SUMMARY_LOG"
            echo "使用命令：" | tee -a "$SUMMARY_LOG"
            echo "  USE_XLA=1 AMP_MODE=bf16 XLA_COMPILE_MODE=parallel \\" | tee -a "$SUMMARY_LOG"
            echo "    TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 \\" | tee -a "$SUMMARY_LOG"
            echo "    ./run_optimized.sh 100 1536 'xla_bf16_final' 1" | tee -a "$SUMMARY_LOG"
            BEST_CONFIG="xla_bf16"
        else
            echo "=========================================" | tee -a "$SUMMARY_LOG"
            echo "⚠️ 测试3 失败：XLA + BF16 不稳定" | tee -a "$SUMMARY_LOG"
            echo "=========================================" | tee -a "$SUMMARY_LOG"
            echo "推荐配置：XLA + FP32" | tee -a "$SUMMARY_LOG"
            echo "" | tee -a "$SUMMARY_LOG"
            echo "使用命令：" | tee -a "$SUMMARY_LOG"
            echo "  USE_XLA=1 AMP_MODE=off XLA_COMPILE_MODE=parallel \\" | tee -a "$SUMMARY_LOG"
            echo "    ./run_optimized.sh 100 1536 'xla_fp32_final' 1" | tee -a "$SUMMARY_LOG"
            BEST_CONFIG="xla_fp32"
        fi
    else
        echo "=========================================" | tee -a "$SUMMARY_LOG"
        echo "⚠️ 测试2 失败：XLA 在 FP32 下不稳定" | tee -a "$SUMMARY_LOG"
        echo "=========================================" | tee -a "$SUMMARY_LOG"
        echo "推荐配置：无 XLA + BF16 + TF32" | tee -a "$SUMMARY_LOG"
        echo "" | tee -a "$SUMMARY_LOG"
        echo "使用命令：" | tee -a "$SUMMARY_LOG"
        echo "  USE_XLA=0 AMP_MODE=bf16 \\" | tee -a "$SUMMARY_LOG"
        echo "    TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 \\" | tee -a "$SUMMARY_LOG"
        echo "    ./run_optimized.sh 100 1536 'no_xla_bf16_final' 1" | tee -a "$SUMMARY_LOG"
        BEST_CONFIG="no_xla_bf16"
    fi
else
    echo "=========================================" | tee -a "$SUMMARY_LOG"
    echo "❌ 测试1 失败：基础配置存在问题" | tee -a "$SUMMARY_LOG"
    echo "=========================================" | tee -a "$SUMMARY_LOG"
    echo "请检查以下内容：" | tee -a "$SUMMARY_LOG"
    echo "1. 代码修改是否正确" | tee -a "$SUMMARY_LOG"
    echo "2. TensorFlow 是否正确安装" | tee -a "$SUMMARY_LOG"
    echo "3. GPU 驱动是否正常" | tee -a "$SUMMARY_LOG"
    BEST_CONFIG="failed"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "测试完成" | tee -a "$SUMMARY_LOG"
echo "=========================================" | tee -a "$SUMMARY_LOG"
echo "测试结果目录: $TEST_DIR" | tee -a "$SUMMARY_LOG"
echo "测试摘要: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# 生成详细分析报告
REPORT="${TEST_DIR}/analysis_report.md"
cat > "$REPORT" << EOF
# 稳定性测试分析报告

## 测试概况

- **测试时间**: $(date)
- **测试目录**: $TEST_DIR
- **最佳配置**: $BEST_CONFIG

## 测试结果汇总

EOF

# 添加每个测试的结果
for i in 1 2 3; do
    if [ -f "${TEST_DIR}/test${i}_"*".log" ]; then
        test_name=$(ls "${TEST_DIR}/test${i}_"*".log" 2>/dev/null | head -1 | sed 's/.*test[0-9]_\(.*\)\.log/\1/')
        if [ -f "${TEST_DIR}/test${i}_${test_name}_error.log" ]; then
            echo "### 测试${i}: ${test_name} - ❌ 失败" >> "$REPORT"
            echo "" >> "$REPORT"
            echo "**错误信息:**" >> "$REPORT"
            echo '```' >> "$REPORT"
            tail -30 "${TEST_DIR}/test${i}_${test_name}_error.log" >> "$REPORT" 2>/dev/null || echo "无错误日志" >> "$REPORT"
            echo '```' >> "$REPORT"
        else
            echo "### 测试${i}: ${test_name} - ✅ 成功" >> "$REPORT"
            echo "" >> "$REPORT"
            if [ -f "${TEST_DIR}/test${i}_${test_name}_timing.txt" ]; then
                echo "**性能数据:**" >> "$REPORT"
                echo '```' >> "$REPORT"
                cat "${TEST_DIR}/test${i}_${test_name}_timing.txt" >> "$REPORT" 2>/dev/null || echo "无性能数据" >> "$REPORT"
                echo '```' >> "$REPORT"
            fi
        fi
        echo "" >> "$REPORT"
    fi
done

# 添加推荐配置
cat >> "$REPORT" << EOF

## 推荐配置

根据测试结果，推荐使用以下配置：

EOF

case "$BEST_CONFIG" in
    "xla_bf16")
        cat >> "$REPORT" << 'EOF'
### ✅ XLA + BF16 + TF32（最快）

```bash
USE_XLA=1 AMP_MODE=bf16 XLA_COMPILE_MODE=parallel \
  TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 NVIDIA_TF32_OVERRIDE=1 \
  ./run_optimized.sh 100 1536 'production_xla_bf16' 1
```

**预期性能**: 40-60秒/回合
**稳定性**: 高
EOF
        ;;
    "xla_fp32")
        cat >> "$REPORT" << 'EOF'
### ⚠️ XLA + FP32（次优）

```bash
USE_XLA=1 AMP_MODE=off XLA_COMPILE_MODE=parallel \
  ./run_optimized.sh 100 1536 'production_xla_fp32' 1
```

**预期性能**: 60-70秒/回合
**稳定性**: 中等
**注意**: 显存占用是 BF16 的 2 倍
EOF
        ;;
    "no_xla_bf16")
        cat >> "$REPORT" << 'EOF'
### ⚠️ 无 XLA + BF16 + TF32（保守）

```bash
USE_XLA=0 AMP_MODE=bf16 \
  TF_ENABLE_CUBLAS_TF32=1 TF_USE_CUDNN_TF32=1 NVIDIA_TF32_OVERRIDE=1 \
  ./run_optimized.sh 100 1536 'production_no_xla' 1
```

**预期性能**: 70-80秒/回合
**稳定性**: 非常高
**适用场景**: XLA 在当前硬件上不稳定时使用
EOF
        ;;
    "failed")
        cat >> "$REPORT" << 'EOF'
### ❌ 所有测试失败

请检查：
1. TensorFlow 安装是否正确
2. GPU 驱动版本是否兼容
3. CUDA 版本是否匹配
4. 代码修改是否引入新问题

查看详细错误日志：`test1_no_xla_bf16_tf32_error.log`
EOF
        ;;
esac

cat >> "$REPORT" << EOF

## 详细日志文件

EOF

# 列出所有日志文件
for log_file in "${TEST_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        echo "- \`$(basename "$log_file")\`" >> "$REPORT"
    fi
done

echo "" >> "$REPORT"
echo "## 文件说明" >> "$REPORT"
echo "" >> "$REPORT"
echo "- \`test_summary.log\`: 测试摘要（包含所有测试的概况）" >> "$REPORT"
echo "- \`test{N}_{name}.log\`: 完整的训练日志" >> "$REPORT"
echo "- \`test{N}_{name}_error.log\`: 提取的错误信息" >> "$REPORT"
echo "- \`test{N}_{name}_timing.txt\`: 性能统计数据" >> "$REPORT"
echo "- \`analysis_report.md\`: 本分析报告" >> "$REPORT"

echo "" | tee -a "$SUMMARY_LOG"
echo "详细分析报告已生成: $REPORT" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
echo "查看报告:" | tee -a "$SUMMARY_LOG"
echo "  cat $REPORT" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"
