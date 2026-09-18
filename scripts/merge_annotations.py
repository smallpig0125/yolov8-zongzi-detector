from pathlib import Path
import ast
import re


# ============================================================
# 基本設定
# ============================================================

# 程式預設放在 dataset 資料夾內
DATASET_DIR = Path(__file__).resolve().parent

# 輸出資料夾
OUTPUT_DIR = DATASET_DIR / "merged_annotations_by_class"

# 是否包含 test 資料
INCLUDE_TEST = False

# 是否在輸出檔中加入來源檔案註解
# False：純合併標註內容
# True ：每列標註前會加上 # Source: xxx.txt
ADD_SOURCE_HEADER = False

# Roboflow / YOLO labels 結構下，是否移除每列最前面的 class_id
# False：保留完整 YOLO 標註，例如：0 0.5 0.5 0.1 0.1
# True ：只保留座標，例如：0.5 0.5 0.1 0.1
REMOVE_CLASS_ID_IN_OUTPUT = False


# ============================================================
# 工具函式
# ============================================================

def sanitize_filename(name):
    """
    將類別名稱轉成可作為檔名的安全格式。
    """
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name if name else "unknown_class"


def parse_names_from_data_yaml(yaml_path):
    """
    讀取 data.yaml 中的 names 欄位。

    支援以下常見格式：

    names: ['car', 'bus', 'truck']

    names:
      - car
      - bus
      - truck

    names:
      0: car
      1: bus
      2: truck
    """
    if not yaml_path.exists():
        return {}

    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    names_line_index = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("names:"):
            names_line_index = i
            break

    if names_line_index is None:
        return {}

    first_line = lines[names_line_index].strip()
    after_colon = first_line.split(":", 1)[1].strip()

    # 格式：names: ['car', 'bus']
    if after_colon:
        try:
            parsed = ast.literal_eval(after_colon)

            if isinstance(parsed, list):
                return {i: str(name) for i, name in enumerate(parsed)}

            if isinstance(parsed, dict):
                return {int(k): str(v) for k, v in parsed.items()}

        except Exception:
            pass

    # 多行格式
    collected_list = []
    collected_dict = {}

    for line in lines[names_line_index + 1:]:
        # 遇到下一個 top-level 欄位就停止
        if line and not line.startswith((" ", "\t", "-")) and ":" in line:
            break

        stripped = line.strip()

        if not stripped:
            continue

        # 格式：
        # names:
        #   - car
        #   - bus
        if stripped.startswith("-"):
            value = stripped[1:].strip().strip("'\"")
            collected_list.append(value)
            continue

        # 格式：
        # names:
        #   0: car
        #   1: bus
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")

            if key.isdigit():
                collected_dict[int(key)] = value

    if collected_dict:
        return collected_dict

    if collected_list:
        return {i: name for i, name in enumerate(collected_list)}

    return {}


def get_class_name_from_id(class_id, class_names):
    """
    依照 class_id 取得類別名稱。
    若 data.yaml 找不到對應名稱，則使用 class_0、class_1。
    """
    if class_id in class_names:
        return class_names[class_id]

    return f"class_{class_id}"


# ============================================================
# 結構偵測
# ============================================================

def detect_yolo_or_roboflow_label_dirs():
    """
    偵測 Roboflow / Ultralytics YOLO 類型的 labels 資料夾。

    支援：
    dataset/train/labels
    dataset/valid/labels
    dataset/val/labels
    dataset/test/labels

    以及：
    dataset/labels/train
    dataset/labels/valid
    dataset/labels/val
    dataset/labels/test
    """
    split_names = ["train", "valid", "val"]

    if INCLUDE_TEST:
        split_names.append("test")

    candidate_dirs = []

    # Roboflow 常見結構：train/labels、valid/labels
    for split in split_names:
        label_dir = DATASET_DIR / split / "labels"
        if label_dir.exists() and label_dir.is_dir():
            if list(label_dir.rglob("*.txt")):
                candidate_dirs.append(label_dir)

    # Ultralytics 常見結構：labels/train、labels/val
    for split in split_names:
        label_dir = DATASET_DIR / "labels" / split
        if label_dir.exists() and label_dir.is_dir():
            if list(label_dir.rglob("*.txt")):
                candidate_dirs.append(label_dir)

    # 去除重複
    unique_dirs = []
    seen = set()

    for d in candidate_dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(d)

    return unique_dirs


def detect_class_folder_dirs():
    """
    偵測一般類別資料夾結構。

    例如：
    dataset/train/car/*.txt
    dataset/train/bus/*.txt
    dataset/val/car/*.txt
    dataset/val/bus/*.txt
    """
    split_names = ["train", "valid", "val"]

    if INCLUDE_TEST:
        split_names.append("test")

    ignored_dir_names = {
        "images",
        "labels",
        "image",
        "label",
        "__pycache__"
    }

    class_dirs = []

    for split in split_names:
        split_dir = DATASET_DIR / split

        if not split_dir.exists() or not split_dir.is_dir():
            continue

        for subdir in sorted(split_dir.iterdir()):
            if not subdir.is_dir():
                continue

            if subdir.name.lower() in ignored_dir_names:
                continue

            txt_files = list(subdir.rglob("*.txt"))

            if txt_files:
                class_dirs.append({
                    "split": split,
                    "class_name": subdir.name,
                    "dir": subdir
                })

    return class_dirs


def auto_detect_dataset_type():
    """
    自動判斷目前資料集格式。

    優先順序：
    1. Roboflow / Ultralytics YOLO labels 結構
    2. train/val 類別資料夾結構

    這樣可以避免把 Roboflow 的 train/images、train/labels 誤判成類別。
    """
    yolo_label_dirs = detect_yolo_or_roboflow_label_dirs()

    if yolo_label_dirs:
        return "yolo_or_roboflow", yolo_label_dirs

    class_folder_dirs = detect_class_folder_dirs()

    if class_folder_dirs:
        return "class_folder", class_folder_dirs

    return "unknown", []


# ============================================================
# 合併模式 1：Roboflow / Ultralytics YOLO labels 結構
# ============================================================

def merge_yolo_or_roboflow_annotations(label_dirs, class_names):
    """
    讀取所有 labels/*.txt，依照每列第一欄 class_id 分類合併。
    """
    merged = {}

    total_files = 0
    total_lines = 0
    skipped_lines = 0

    for label_dir in label_dirs:
        txt_files = sorted(label_dir.rglob("*.txt"))

        for txt_file in txt_files:
            total_files += 1

            relative_path = txt_file.relative_to(DATASET_DIR)

            with txt_file.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, start=1):
                raw_line = line.strip()

                if not raw_line:
                    continue

                if raw_line.startswith("#"):
                    continue

                parts = raw_line.split()

                if len(parts) < 2:
                    skipped_lines += 1
                    print(f"[跳過] 格式欄位不足：{relative_path} 第 {line_no} 行")
                    continue

                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    skipped_lines += 1
                    print(f"[跳過] class_id 無法解析：{relative_path} 第 {line_no} 行：{raw_line}")
                    continue

                class_name = get_class_name_from_id(class_id, class_names)
                safe_class_name = sanitize_filename(class_name)

                if safe_class_name not in merged:
                    merged[safe_class_name] = []

                if ADD_SOURCE_HEADER:
                    merged[safe_class_name].append(f"# Source: {relative_path}, line {line_no}\n")

                if REMOVE_CLASS_ID_IN_OUTPUT:
                    output_line = " ".join(parts[1:])
                else:
                    output_line = raw_line

                merged[safe_class_name].append(output_line + "\n")
                total_lines += 1

    for class_name, lines in merged.items():
        output_file = OUTPUT_DIR / f"{class_name}.txt"

        with output_file.open("w", encoding="utf-8") as f:
            f.writelines(lines)

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "skipped_lines": skipped_lines,
        "class_count": len(merged)
    }


# ============================================================
# 合併模式 2：train/val 類別資料夾結構
# ============================================================

def merge_class_folder_annotations(class_folder_dirs):
    """
    依照 train/val 底下的類別資料夾名稱合併 txt。
    """
    merged = {}

    total_files = 0
    total_lines = 0

    for item in class_folder_dirs:
        class_name = item["class_name"]
        class_dir = item["dir"]
        safe_class_name = sanitize_filename(class_name)

        if safe_class_name not in merged:
            merged[safe_class_name] = []

        txt_files = sorted(class_dir.rglob("*.txt"))

        for txt_file in txt_files:
            total_files += 1
            relative_path = txt_file.relative_to(DATASET_DIR)

            with txt_file.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if ADD_SOURCE_HEADER:
                merged[safe_class_name].append(f"# Source: {relative_path}\n")

            for line in lines:
                raw_line = line.strip()

                if not raw_line:
                    continue

                merged[safe_class_name].append(raw_line + "\n")
                total_lines += 1

    for class_name, lines in merged.items():
        output_file = OUTPUT_DIR / f"{class_name}.txt"

        with output_file.open("w", encoding="utf-8") as f:
            f.writelines(lines)

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "class_count": len(merged)
    }


# ============================================================
# 主程式
# ============================================================

def main():
    print("========================================")
    print("自動合併 YOLO / Roboflow 註解檔")
    print("========================================")
    print(f"資料集資料夾：{DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_yaml_path = DATASET_DIR / "data.yaml"
    class_names = parse_names_from_data_yaml(data_yaml_path)

    if class_names:
        print(f"\n已讀取 data.yaml 類別數：{len(class_names)}")
        for class_id, class_name in sorted(class_names.items()):
            print(f"  {class_id}: {class_name}")
    else:
        print("\n未讀取到 data.yaml 類別名稱。")
        print("若為 YOLO / Roboflow labels 結構，輸出檔名將使用 class_0、class_1、class_2...")

    dataset_type, detected_items = auto_detect_dataset_type()

    print("\n偵測資料集結構：")

    if dataset_type == "yolo_or_roboflow":
        print("模式：Roboflow / Ultralytics YOLO labels 結構")
        print("找到 labels 資料夾：")

        for d in detected_items:
            print(f"  - {d.relative_to(DATASET_DIR)}")

        result = merge_yolo_or_roboflow_annotations(detected_items, class_names)

        print("\n合併完成。")
        print(f"處理 txt 檔案數：{result['total_files']}")
        print(f"合併標註列數：{result['total_lines']}")
        print(f"跳過異常列數：{result['skipped_lines']}")
        print(f"輸出類別數：{result['class_count']}")
        print(f"輸出資料夾：{OUTPUT_DIR}")

    elif dataset_type == "class_folder":
        print("模式：train / val 類別資料夾結構")
        print("找到類別資料夾：")

        for item in detected_items:
            rel_dir = item["dir"].relative_to(DATASET_DIR)
            print(f"  - {rel_dir}")

        result = merge_class_folder_annotations(detected_items)

        print("\n合併完成。")
        print(f"處理 txt 檔案數：{result['total_files']}")
        print(f"合併標註列數：{result['total_lines']}")
        print(f"輸出類別數：{result['class_count']}")
        print(f"輸出資料夾：{OUTPUT_DIR}")

    else:
        print("無法辨識資料集結構。")
        print("\n支援的結構包含：")
        print("1. Roboflow：dataset/train/labels、dataset/valid/labels")
        print("2. Ultralytics：dataset/labels/train、dataset/labels/val")
        print("3. 類別資料夾：dataset/train/class_name/*.txt、dataset/val/class_name/*.txt")
        return

    print("\n輸出檔案：")
    for output_file in sorted(OUTPUT_DIR.glob("*.txt")):
        print(f"  - {output_file.relative_to(DATASET_DIR)}")


if __name__ == "__main__":
    main()