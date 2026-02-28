#!/bin/bash
# Quick Fix v6.3 - 修复 v6.2 性能下降问题
#
# 改动：
# 1. ✅ 禁用 Dropout (dropout_rate=0.0)
# 2. ✅ 恢复简单奖励 (移除 ±0.005 微调)
# 3. 🔧 禁用课程学习（使用固定难度）

echo "=================================================="
echo "v6.3 快速修复训练"
echo "目标：恢复到 v6.1 的 35-40 平均分"
echo "=================================================="
echo ""
echo "改动说明："
echo "  ✅ Dropout 已禁用 (0.0)"
echo "  ✅ 奖励函数已简化"
echo "  🔧 将使用固定难度训练"
echo ""
echo "开始训练 500 轮（快速验证）..."
echo ""

python train.py --episodes 500 --no-curriculum

echo ""
echo "=================================================="
echo "训练完成！"
echo ""
echo "检查结果："
echo "  1. 查看 model/training_curves.png"
echo "  2. 期望平均分：25-30（500轮）"
echo "  3. 如果成功，继续训练 1000 轮"
echo ""
echo "下一步："
echo "  如果效果好：python train.py --episodes 1000 --no-curriculum"
echo "  如果仍然差：完全回退到 v6.1"
echo "=================================================="
