#!/bin/bash
# RefCOCO/RefCOCOg 数据准备脚本
#
# 数据来源：通过 HuggingFace datasets 下载（内网有镜像加速）
# - COCO 2014 Train images
# - RefCOCO annotations
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
# └── text_features.pt     # 预计算的文本特征

# NFS 持久化路径（训练时读取）
NFS_ROOT="/primus_datasets/external_data/edu/mllm/fwk/vlm"
NFS_DATA_ROOT="$NFS_ROOT/refcoco"

# 本地临时路径（下载/解压用，NFS 不支持直接写入）
LOCAL_ROOT="/root/storage/vlm_data"
LOCAL_DATA_ROOT="$LOCAL_ROOT/refcoco"

mkdir -p "$LOCAL_DATA_ROOT/images"
mkdir -p "$NFS_DATA_ROOT"

echo "============================================"
echo "  存储路径说明"
echo "  本地临时: $LOCAL_DATA_ROOT"
echo "  NFS 持久: $NFS_DATA_ROOT"
echo "============================================"

# 安装依赖
echo ""
echo "============================================"
echo "  Step 0: 安装依赖"
echo "============================================"
pip install datasets huggingface_hub pillow 2>/dev/null || true

echo ""
echo "============================================"
echo "  Step 1-3: 通过 HuggingFace 下载数据"
echo "============================================"

# 使用 Python 脚本统一处理下载逻辑
python3 << 'PYTHON_SCRIPT'
import os
import json
import pickle
import sys

LOCAL_DATA_ROOT = "/root/storage/vlm_data/refcoco"
NFS_DATA_ROOT = "/primus_datasets/external_data/edu/mllm/fwk/vlm/refcoco"

os.makedirs(os.path.join(LOCAL_DATA_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(NFS_DATA_ROOT, "images"), exist_ok=True)

# ---- Step 1: 下载 COCO 图像 ----
nfs_images_dir = os.path.join(NFS_DATA_ROOT, "images")
has_images = os.path.isdir(nfs_images_dir) and len(os.listdir(nfs_images_dir)) > 1000

if not has_images:
    print("=" * 50)
    print("  下载 COCO 2014 Train Images (via HuggingFace)")
    print("  这可能需要一些时间...")
    print("=" * 50)

    try:
        from datasets import load_dataset

        # 使用 HuggingFace 的 COCO 数据集
        # 尝试多个可用的 COCO 数据源
        dataset = None
        hf_repos = [
            ("detection-datasets/coco", "train", None),
            ("HuggingFaceM4/COCO", "train", "2014"),
        ]

        for repo_id, split, config in hf_repos:
            try:
                print(f"尝试从 {repo_id} 下载...")
                if config:
                    dataset = load_dataset(repo_id, config, split=split, trust_remote_code=True)
                else:
                    dataset = load_dataset(repo_id, split=split, trust_remote_code=True)
                print(f"成功！数据集大小: {len(dataset)}")
                break
            except Exception as e:
                print(f"  {repo_id} 失败: {e}")
                continue

        if dataset is not None:
            # 保存图像到本地，再复制到 NFS
            local_images_dir = os.path.join(LOCAL_DATA_ROOT, "images")
            print(f"保存图像到本地: {local_images_dir}")

            # 同时构建 instances.json
            annotations = []
            images_info = []
            saved_count = 0

            for idx, item in enumerate(dataset):
                # 不同数据集的字段名可能不同
                image = item.get("image", item.get("img", None))
                image_id = item.get("image_id", item.get("id", idx))

                if image is None:
                    continue

                # 保存图像
                filename = f"COCO_train2014_{image_id:012d}.jpg"
                local_path = os.path.join(local_images_dir, filename)

                if not os.path.exists(local_path):
                    image.save(local_path)

                images_info.append({
                    "id": image_id,
                    "file_name": filename,
                    "width": image.width,
                    "height": image.height,
                })

                # 提取 annotations（如果有）
                objects = item.get("objects", item.get("annotations", None))
                if objects and isinstance(objects, dict):
                    bboxes = objects.get("bbox", [])
                    categories = objects.get("category", objects.get("label", []))
                    for i, bbox in enumerate(bboxes):
                        ann_id = len(annotations)
                        cat_id = categories[i] if i < len(categories) else 0
                        annotations.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                        })

                saved_count += 1
                if saved_count % 5000 == 0:
                    print(f"  已保存 {saved_count} 张图像...")

            print(f"共保存 {saved_count} 张图像")

            # 保存 instances.json
            if annotations:
                instances = {
                    "images": images_info,
                    "annotations": annotations,
                    "categories": [],
                }
                local_instances = os.path.join(LOCAL_DATA_ROOT, "instances.json")
                with open(local_instances, "w") as f:
                    json.dump(instances, f)
                print(f"instances.json 已保存: {len(annotations)} annotations")

            # 复制到 NFS
            print("复制图像到 NFS（这可能需要较长时间）...")
            import shutil
            # 逐个复制，避免一次性复制太多文件
            nfs_img_dir = os.path.join(NFS_DATA_ROOT, "images")
            local_files = os.listdir(local_images_dir)
            for i, fname in enumerate(local_files):
                src = os.path.join(local_images_dir, fname)
                dst = os.path.join(nfs_img_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                if (i + 1) % 10000 == 0:
                    print(f"  已复制 {i+1}/{len(local_files)} 张到 NFS...")
            print(f"图像复制完成: {len(local_files)} 张")

            # 复制 instances.json
            if os.path.exists(os.path.join(LOCAL_DATA_ROOT, "instances.json")):
                shutil.copy2(
                    os.path.join(LOCAL_DATA_ROOT, "instances.json"),
                    os.path.join(NFS_DATA_ROOT, "instances.json"),
                )
                print("instances.json 已复制到 NFS")
        else:
            print("⚠️ 所有 HuggingFace COCO 数据源都失败了")
            print("请手动下载 COCO train2014 图像并放到:")
            print(f"  {NFS_DATA_ROOT}/images/")
            sys.exit(1)

    except ImportError:
        print("⚠️ datasets 库未安装，请运行: pip install datasets")
        sys.exit(1)
else:
    print(f"COCO images 已存在于 NFS ({nfs_images_dir})，跳过下载")

# ---- Step 2: 下载 RefCOCO annotations ----
refs_path = os.path.join(NFS_DATA_ROOT, "refs(unc).p")
if not os.path.exists(refs_path):
    print("\n" + "=" * 50)
    print("  下载 RefCOCO Annotations (via HuggingFace)")
    print("=" * 50)

    import shutil
    local_refs = os.path.join(LOCAL_DATA_ROOT, "refs(unc).p")

    try:
        from huggingface_hub import hf_hub_download

        # 尝试从 HuggingFace 下载 RefCOCO
        hf_sources = [
            ("lichengunc/refer", "refcoco/refs(unc).p", "dataset"),
            ("jxu124/refcoco", "refs(unc).p", "dataset"),
        ]

        downloaded = False
        for repo_id, filename, repo_type in hf_sources:
            try:
                print(f"尝试从 {repo_id} 下载 {filename}...")
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                )
                shutil.copy2(path, local_refs)
                print(f"下载成功: {path}")
                downloaded = True
                break
            except Exception as e:
                print(f"  失败: {e}")
                continue

        if downloaded:
            shutil.copy2(local_refs, refs_path)
            print(f"refs(unc).p 已复制到 NFS: {refs_path}")

            # 验证
            with open(refs_path, "rb") as f:
                refs = pickle.load(f)
            total_sents = sum(len(r.get("sentences", [])) for r in refs)
            print(f"RefCOCO: {len(refs)} refs, {total_sents} sentences")
        else:
            print("⚠️ RefCOCO annotations 下载失败")
            print("请手动下载并放到:")
            print(f"  {refs_path}")

    except ImportError:
        print("⚠️ huggingface_hub 未安装，请运行: pip install huggingface_hub")
else:
    print(f"\nRefCOCO annotations 已存在于 NFS，跳过下载")

print("\n数据下载步骤完成！")
PYTHON_SCRIPT

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
    echo "复制到 NFS..."
    cp "$LOCAL_DATA_ROOT/text_features.pt" "$NFS_DATA_ROOT/text_features.pt"
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
ls -la "$NFS_DATA_ROOT/" 2>/dev/null || echo "  (NFS 目录暂不可访问)"
echo ""
echo "下一步：运行预训练"
echo "  python3 scripts/run_pretrain.py --data_root $NFS_DATA_ROOT"
