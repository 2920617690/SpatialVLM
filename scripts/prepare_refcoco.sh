#!/bin/bash
# RefCOCO/RefCOCOg 数据准备脚本
#
# 数据来源：
# - COCO 2014 Train images（约 13GB）
# - RefCOCO/RefCOCOg annotations（来自 lichengunc/refer）
#
# 目录结构（准备完成后）：
# data/refcoco/
# ├── images/              # COCO train2014 图像
# ├── instances.json       # COCO annotations
# ├── refs(unc).p          # RefCOCO referring expressions
# └── text_features.pt     # 预计算的文本特征（由 precompute_text_features.py 生成）

set -e

DATA_ROOT="data/refcoco"
mkdir -p "$DATA_ROOT/images"

echo "============================================"
echo "  Step 1: 下载 COCO 2014 Train Images"
echo "============================================"

COCO_URL="http://images.cocodataset.org/zips/train2014.zip"
COCO_ZIP="$DATA_ROOT/train2014.zip"

if [ ! -f "$COCO_ZIP" ] && [ ! -d "$DATA_ROOT/images/COCO_train2014_000000000009.jpg" ]; then
    echo "下载 COCO train2014 images..."
    echo "文件较大（约 13GB），请耐心等待..."
    wget -c "$COCO_URL" -O "$COCO_ZIP"
    echo "解压中..."
    unzip -q "$COCO_ZIP" -d "$DATA_ROOT/"
    # 将图片移到 images/ 目录
    mv "$DATA_ROOT/train2014/"* "$DATA_ROOT/images/" 2>/dev/null || true
    rmdir "$DATA_ROOT/train2014" 2>/dev/null || true
    rm -f "$COCO_ZIP"
    echo "COCO images 准备完成"
else
    echo "COCO images 已存在，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 2: 下载 COCO Annotations"
echo "============================================"

ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANN_ZIP="$DATA_ROOT/annotations_trainval2014.zip"

if [ ! -f "$DATA_ROOT/instances.json" ]; then
    echo "下载 COCO annotations..."
    wget -c "$ANN_URL" -O "$ANN_ZIP"
    unzip -q "$ANN_ZIP" -d "$DATA_ROOT/"
    # 提取 instances_train2014.json
    cp "$DATA_ROOT/annotations/instances_train2014.json" "$DATA_ROOT/instances.json"
    rm -rf "$DATA_ROOT/annotations" "$ANN_ZIP"
    echo "COCO annotations 准备完成"
else
    echo "COCO annotations 已存在，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 3: 下载 RefCOCO Annotations"
echo "============================================"

# 从 lichengunc/refer 仓库获取 RefCOCO annotations
REFER_URL="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
REFCOCO_ZIP="$DATA_ROOT/refcoco.zip"

if [ ! -f "$DATA_ROOT/refs(unc).p" ]; then
    echo "下载 RefCOCO annotations..."
    wget -c "$REFER_URL" -O "$REFCOCO_ZIP" || {
        echo "直接下载失败，尝试从 HuggingFace 获取..."
        # 备用方案：从 HuggingFace datasets 获取
        python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='lichengunc/refer',
    filename='refcoco/refs(unc).p',
    repo_type='dataset',
)
shutil.copy(path, '$DATA_ROOT/refs(unc).p')
print('从 HuggingFace 下载成功')
" || {
            echo ""
            echo "⚠️  自动下载失败。请手动下载 RefCOCO annotations："
            echo "  1. 访问 https://github.com/lichengunc/refer"
            echo "  2. 下载 refcoco/refs(unc).p"
            echo "  3. 放到 $DATA_ROOT/refs(unc).p"
            echo ""
        }
    }

    if [ -f "$REFCOCO_ZIP" ]; then
        unzip -q "$REFCOCO_ZIP" -d "$DATA_ROOT/"
        # 移动文件到正确位置
        if [ -d "$DATA_ROOT/refcoco" ]; then
            mv "$DATA_ROOT/refcoco/"* "$DATA_ROOT/" 2>/dev/null || true
            rmdir "$DATA_ROOT/refcoco" 2>/dev/null || true
        fi
        rm -f "$REFCOCO_ZIP"
    fi

    echo "RefCOCO annotations 准备完成"
else
    echo "RefCOCO annotations 已存在，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 4: 预计算文本特征"
echo "============================================"

if [ ! -f "$DATA_ROOT/text_features.pt" ]; then
    echo "运行文本特征预计算脚本..."
    echo "（需要 GPU，首次运行会下载 SigLIP text encoder）"
    python3 scripts/precompute_text_features.py \
        --data_root "$DATA_ROOT" \
        --output "$DATA_ROOT/text_features.pt" \
        --batch_size 256
    echo "文本特征预计算完成"
else
    echo "文本特征已存在，跳过预计算"
fi

echo ""
echo "============================================"
echo "  数据准备完成！"
echo "============================================"
echo ""
echo "目录结构："
ls -la "$DATA_ROOT/"
echo ""
echo "下一步：运行预训练"
echo "  python3 scripts/run_pretrain.py"
