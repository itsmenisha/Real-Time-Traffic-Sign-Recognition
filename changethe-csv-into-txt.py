import os
import pandas as pd

# Paths
csv_path = r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\GTSRB\train.csv"
output_base = r"C:\Users\Acer\Desktop\projects\Real-Time Traffic Sign Recognition\data\GTSRB\trainvalues"
os.makedirs(output_base, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    img_width, img_height = row["Width"], row["Height"]
    x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
    class_id = row["ClassId"]
    image_path = row["Path"]

    # Compute YOLO format
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    x_center = x1 + bbox_width / 2
    y_center = y1 + bbox_height / 2

    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height

    # Folder per class (optional)
    class_folder = os.path.join(output_base, str(class_id))
    os.makedirs(class_folder, exist_ok=True)

    # TXT file name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_filename = os.path.join(class_folder, base_name + ".txt")

    # Write YOLO label
    with open(txt_filename, "w") as f:
        f.write(
            f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

print("✅ YOLO txt labels created and normalized.")


