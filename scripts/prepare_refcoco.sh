#!/bin/bash
# RefCOCO/RefCOCOg 数据准备脚本
#
# 数据来源：
# - COCO 2014 Train images（约 13GB）
# - RefCOCO/RefCOCOg annotations（来自 lichengunc/refer）
#
# 存储策略（Primus NFS）：
# - 先下载到本地 /root/storage（NFS 不支持直接写入大文件）
# - 再复制到 NFS 持久化路径 /primus_datasets/external_data/edu/mllm/fwk/vlm/
# - 训练时从 NFS 路径读取
#
# 最终目录结构：
# /primus_datasets/external_data/edu/mllm/fwk/vlm/refcoco/
# ├── images/              # COCO train2014 图像
# ├── instances.json       # COCO annotations
# ├── refs(unc).p          # RefCOCO referring expressions
# └── text_features.pt     # 预计算的文本特征（由 precompute_text_features.py 生成）

# NFS 持久化路径（训练时读取）
NFS_ROOT="/primus_datasets/external_data/edu/mllm/fwk/vlm"
NFS_DATA_ROOT="$NFS_ROOT/refcoco"

# 本地临时路径（下载/解压用，NFS 不支持直接写入）
LOCAL_ROOT="/root/storage/vlm_data"
LOCAL_DATA_ROOT="$LOCAL_ROOT/refcoco"

mkdir -p "$LOCAL_DATA_ROOT/images"
mkdir -p "$NFS_DATA_ROOT"

# 辅助函数：将本地文件复制到 NFS
copy_to_nfs() {
    local src="$1"
    local dst="$2"
    echo "复制到 NFS: $src → $dst"
    cp -r "$src" "$dst"
    echo "复制完成"
}

echo "============================================"
echo "  存储路径说明"
echo "  本地临时: $LOCAL_DATA_ROOT"
echo "  NFS 持久: $NFS_DATA_ROOT"
echo "============================================"

echo ""
echo "============================================"
echo "  Step 1: 下载 COCO 2014 Train Images"
echo "============================================"

COCO_URL="http://images.cocodataset.org/zips/train2014.zip"
COCO_ZIP="$LOCAL_DATA_ROOT/train2014.zip"

if [ ! -d "$NFS_DATA_ROOT/images" ] || [ -z "$(ls -A $NFS_DATA_ROOT/images 2>/dev/null)" ]; then
    echo "下载 COCO train2014 images..."
    echo "文件较大（约 13GB），请耐心等待..."
    wget -c "$COCO_URL" -O "$COCO_ZIP"
    echo "解压中..."
    unzip -q "$COCO_ZIP" -d "$LOCAL_DATA_ROOT/"
    # 将图片移到 images/ 目录
    mv "$LOCAL_DATA_ROOT/train2014/"* "$LOCAL_DATA_ROOT/images/" 2>/dev/null || true
    rmdir "$LOCAL_DATA_ROOT/train2014" 2>/dev/null || true
    rm -f "$COCO_ZIP"
    echo "COCO images 下载完成，复制到 NFS..."
    mkdir -p "$NFS_DATA_ROOT/images"
    cp -r "$LOCAL_DATA_ROOT/images/"* "$NFS_DATA_ROOT/images/"
    echo "COCO images 准备完成"
else
    echo "COCO images 已存在于 NFS，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 2: 下载 COCO Annotations"
echo "============================================"

ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANN_ZIP="$LOCAL_DATA_ROOT/annotations_trainval2014.zip"

if [ ! -f "$NFS_DATA_ROOT/instances.json" ]; then
    echo "下载 COCO annotations..."
    wget -c "$ANN_URL" -O "$ANN_ZIP"
    unzip -q "$ANN_ZIP" -d "$LOCAL_DATA_ROOT/"
    # 提取 instances_train2014.json
    cp "$LOCAL_DATA_ROOT/annotations/instances_train2014.json" "$LOCAL_DATA_ROOT/instances.json"
    rm -rf "$LOCAL_DATA_ROOT/annotations" "$ANN_ZIP"
    # 复制到 NFS
    copy_to_nfs "$LOCAL_DATA_ROOT/instances.json" "$NFS_DATA_ROOT/instances.json"
    echo "COCO annotations 准备完成"
else
    echo "COCO annotations 已存在于 NFS，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 3: 下载 RefCOCO Annotations"
echo "============================================"

# 从 lichengunc/refer 仓库获取 RefCOCO annotations
REFER_URL="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
REFCOCO_ZIP="$LOCAL_DATA_ROOT/refcoco.zip"

if [ ! -f "$NFS_DATA_ROOT/refs(unc).p" ]; then
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
shutil.copy(path, '$LOCAL_DATA_ROOT/refs(unc).p')
print('从 HuggingFace 下载成功')
" || {
            echo ""
            echo "⚠️  自动下载失败。请手动下载 RefCOCO annotations："
            echo "  1. 访问 https://github.com/lichengunc/refer"
            echo "  2. 下载 refcoco/refs(unc).p"
            echo "  3. 放到 $NFS_DATA_ROOT/refs(unc).p"
            echo ""
        }
    }

    if [ -f "$REFCOCO_ZIP" ]; then
        unzip -q "$REFCOCO_ZIP" -d "$LOCAL_DATA_ROOT/"
        # 移动文件到正确位置
        if [ -d "$LOCAL_DATA_ROOT/refcoco" ]; then
            mv "$LOCAL_DATA_ROOT/refcoco/"* "$LOCAL_DATA_ROOT/" 2>/dev/null || true
            rmdir "$LOCAL_DATA_ROOT/refcoco" 2>/dev/null || true
        fi
        rm -f "$REFCOCO_ZIP"
    fi

    # 复制到 NFS
    if [ -f "$LOCAL_DATA_ROOT/refs(unc).p" ]; then
        copy_to_nfs "$LOCAL_DATA_ROOT/refs(unc).p" "$NFS_DATA_ROOT/refs(unc).p"
    fi
    echo "RefCOCO annotations 准备完成"
else
    echo "RefCOCO annotations 已存在于 NFS，跳过下载"
fi

echo ""
echo "============================================"
echo "  Step 4: 预计算文本特征"
echo "============================================"

if [ ! -f "$NFS_DATA_ROOT/text_features.pt" ]; then
    echo "运行文本特征预计算脚本..."
    echo "（需要 GPU，首次运行会下载 SigLIP text encoder）"
    # 先输出到本地，再复制到 NFS
    python3 scripts/precompute_text_features.py \
        --data_root "$NFS_DATA_ROOT" \
        --output "$LOCAL_DATA_ROOT/text_features.pt" \
        --batch_size 256
    copy_to_nfs "$LOCAL_DATA_ROOT/text_features.pt" "$NFS_DATA_ROOT/text_features.pt"
    echo "文本特征预计算完成"
else
    echo "文本特征已存在于 NFS，跳过预计算"
fi

echo ""
echo "============================================"
echo "  Step 5: 清理本地临时文件（可选）"
echo "============================================"
echo "本地临时文件位于: $LOCAL_DATA_ROOT"
echo "如需释放本地空间，可手动执行: rm -rf $LOCAL_DATA_ROOT"
echo ""

echo "============================================"
echo "  数据准备完成！"
echo "============================================"
echo ""
echo "NFS 目录结构："
ls -la "$NFS_DATA_ROOT/"
echo ""
echo "下一步：运行预训练"
echo "  python3 scripts/run_pretrain.py --data_root $NFS_DATA_ROOT"
