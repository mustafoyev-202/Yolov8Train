import os
import sys
import yaml
import argparse
from pathlib import Path
import shutil
from glob import glob

def setup_dataset():
    """Setup and prepare the smart office dataset"""
    print("Setting up Smart Office dataset...")
    
    # Create dataset directory structure
    dataset_dir = "smart_office_dataset"
    os.makedirs(f"{dataset_dir}/train/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{dataset_dir}/valid/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/valid/labels", exist_ok=True)
    os.makedirs(f"{dataset_dir}/test/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/test/labels", exist_ok=True)
    
    # Create data.yaml configuration
    data_config = {
        'train': f'{dataset_dir}/train/images',
        'val': f'{dataset_dir}/valid/images', 
        'test': f'{dataset_dir}/test/images',
        'nc': 6,
        'names': ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
    }
    
    with open(f"{dataset_dir}/data.yaml", 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print("Dataset setup complete!")
    return dataset_dir

def download_datasets():
    """Download datasets from Roboflow"""
    try:
        from roboflow import Roboflow
        
        print("Downloading datasets from Roboflow...")
        
        # Roboflow API key (replace with your own)
        rf = Roboflow(api_key="4r5wkLqdTYmymtH2R3Lt")
        
        # Dataset configurations
        datasets = [
            ("person-eccaa-utzc2", 1, "person-1"),
            ("chair-kjohe-ni7vg", 3, "chair-3"), 
            ("test_monitor-hwao9", 1, "test_monitor-1"),
            ("keyboard-v2itg-wq0de", 1, "Keyboard-1"),
            ("laptop-1b8ba-byb6b", 1, "Laptop-1"),
            ("smartphone-f7b9t-fxg03", 1, "smartphone-1")
        ]
        
        for project_id, version, folder_name in datasets:
            print(f"Downloading {project_id}...")
            project = rf.workspace("signlanguage-sbjtq").project(project_id)
            version_obj = project.version(version)
            dataset = version_obj.download("yolov8")
            
        print("All datasets downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        print("Please ensure you have a valid Roboflow API key")

def combine_datasets():
    """Combine downloaded datasets into unified format"""
    print("Combining datasets...")
    
    source_folders = [
        "person-1",
        "chair-3", 
        "test_monitor-1",
        "Keyboard-1",
        "Laptop-1",
        "smartphone-1"
    ]
    
    splits = ["train", "valid", "test"]
    output_base = "smart_office_dataset"
    
    # Create output directories
    for split in splits:
        os.makedirs(f"{output_base}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_base}/{split}/labels", exist_ok=True)
    
    img_extensions = ['.jpg', '.jpeg', '.png']
    
    for source in source_folders:
        for split in splits:
            img_dir = os.path.join(source, split, "images")
            label_dir = os.path.join(source, split, "labels")
            
            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                print(f"Skipping {source}/{split}, folder missing")
                continue
            
            # Copy and rename image files
            for img_path in glob(f"{img_dir}/*"):
                if not any(img_path.endswith(ext) for ext in img_extensions):
                    continue
                base = f"{source}_{os.path.basename(img_path)}"
                out_img_path = os.path.join(output_base, split, "images", base)
                shutil.copy(img_path, out_img_path)
                
                # Copy label file
                label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                label_src_path = os.path.join(label_dir, label_name)
                label_out_path = os.path.join(output_base, split, "labels", os.path.splitext(base)[0] + ".txt")
                
                if os.path.exists(label_src_path):
                    shutil.copy(label_src_path, label_out_path)
    
    print("Dataset combination complete!")

def remap_classes():
    """Remap class IDs to unified format"""
    print("Remapping class IDs...")
    
    target_classes = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
    
    source_class_map = {
        'person-1': 'person',
        'chair-3': 'chair',
        'test_monitor-1': 'monitor',
        'Keyboard-1': 'keyboard',
        'Laptop-1': 'laptop',
        'smartphone-1': 'phone'
    }
    
    dataset_dir = "smart_office_dataset"
    
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_dir, split, "labels")
        for label_file in glob(f"{label_dir}/*.txt"):
            filename = os.path.basename(label_file)
            label_parts = filename.split("_")[0]
            if label_parts not in source_class_map:
                continue
            correct_class = source_class_map[label_parts]
            new_class_id = target_classes.index(correct_class)
            
            # Rewrite label file with correct class ID
            with open(label_file, 'r') as f:
                lines = f.readlines()
            with open(label_file, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = str(new_class_id)
                    f.write(" ".join(parts) + "\n")
    
    print("Class remapping complete!")

def train_model(args):
    """Train the YOLOv8 model"""
    print(f"Starting model training with {args.epochs} epochs...")
    
    # Import ultralytics after setup
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(args.model)
    
    # Train the model
    results = model.train(
        data="smart_office_dataset/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_period=10
    )
    
    print("Training completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Smart Office Object Detection Training")
    parser.add_argument("--download", action="store_true", help="Download datasets from Roboflow")
    parser.add_argument("--setup", action="store_true", help="Setup dataset structure")
    parser.add_argument("--combine", action="store_true", help="Combine downloaded datasets")
    parser.add_argument("--remap", action="store_true", help="Remap class IDs")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--model", default="yolov8m.pt", help="Model to use for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device to use (cpu, 0, 1, etc.)")
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--name", default="smart_office", help="Experiment name")
    
    args = parser.parse_args()
    
    if args.download:
        download_datasets()
    
    if args.setup:
        setup_dataset()
    
    if args.combine:
        combine_datasets()
    
    if args.remap:
        remap_classes()
    
    if args.train:
        train_model(args)
    
    # If no specific action, run full pipeline
    if not any([args.download, args.setup, args.combine, args.remap, args.train]):
        print("Running full training pipeline...")
        download_datasets()
        setup_dataset()
        combine_datasets()
        remap_classes()
        train_model(args)

if __name__ == "__main__":
    main() 