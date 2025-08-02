import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

def evaluate_model(model_path, data_yaml, conf_threshold=0.5, iou_threshold=0.7):
    """Evaluate the trained model"""
    from ultralytics import YOLO
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    print("Running model evaluation...")
    results = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,
        save_txt=True,
        save_conf=True
    )
    
    return results

def generate_evaluation_report(results, output_dir="evaluation_results"):
    """Generate comprehensive evaluation report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    metrics = results.results_dict
    
    # Create detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "model_path": str(results.save_dir),
            "classes": results.names,
            "num_classes": len(results.names)
        },
        "performance_metrics": {
            "mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            "precision": float(metrics.get("metrics/precision(B)", 0)),
            "recall": float(metrics.get("metrics/recall(B)", 0)),
            "f1_score": float(metrics.get("metrics/f1(B)", 0))
        },
        "class_performance": {}
    }
    
    # Extract per-class metrics
    for i, class_name in enumerate(results.names):
        class_metrics = {
            "precision": float(metrics.get(f"metrics/precision(B)/{class_name}", 0)),
            "recall": float(metrics.get(f"metrics/recall(B)/{class_name}", 0)),
            "mAP50": float(metrics.get(f"metrics/mAP50(B)/{class_name}", 0)),
            "mAP50-95": float(metrics.get(f"metrics/mAP50-95(B)/{class_name}", 0))
        }
        report["class_performance"][class_name] = class_metrics
    
    # Save JSON report
    with open(f"{output_dir}/evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate visualizations
    generate_performance_plots(results, output_dir)
    
    print(f"Evaluation report saved to: {output_dir}")
    return report

def generate_performance_plots(results, output_dir):
    """Generate performance visualization plots"""
    
    # 1. Confusion Matrix
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(results.confusion_matrix.matrix, 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=results.names,
                   yticklabels=results.names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Class Performance Bar Chart
    metrics = results.results_dict
    class_names = results.names
    mAP50_scores = []
    
    for class_name in class_names:
        score = metrics.get(f"metrics/mAP50(B)/{class_name}", 0)
        mAP50_scores.append(float(score))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, mAP50_scores, color='skyblue', edgecolor='navy')
    plt.title('mAP50 Scores by Class')
    plt.xlabel('Object Classes')
    plt.ylabel('mAP50 Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, mAP50_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curve
    if hasattr(results, 'pr_curve') and results.pr_curve is not None:
        plt.figure(figsize=(10, 6))
        for i, class_name in enumerate(results.names):
            if i < len(results.pr_curve):
                plt.plot(results.pr_curve[i][0], results.pr_curve[i][1], 
                        label=class_name, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pr_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

def calculate_speed_metrics(model_path, test_images_dir, num_runs=100):
    """Calculate inference speed metrics"""
    from ultralytics import YOLO
    import time
    from glob import glob
    
    print("Calculating speed metrics...")
    model = YOLO(model_path)
    
    # Get test images
    image_files = glob(f"{test_images_dir}/*.jpg") + glob(f"{test_images_dir}/*.png")
    if not image_files:
        print("No test images found!")
        return {}
    
    # Use first image for speed testing
    test_image = image_files[0]
    
    # Warmup
    for _ in range(10):
        _ = model.predict(test_image, verbose=False)
    
    # Speed test
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(test_image, verbose=False)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    speed_metrics = {
        "average_inference_time_ms": avg_time * 1000,
        "std_inference_time_ms": std_time * 1000,
        "fps": fps,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000
    }
    
    print(f"Speed Metrics:")
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Standard deviation: {std_time*1000:.2f} ms")
    
    return speed_metrics

def run_comprehensive_evaluation(model_path, data_yaml, test_images_dir, output_dir="evaluation_results"):
    """Run comprehensive evaluation"""
    print("Starting comprehensive model evaluation...")
    
    # 1. Model performance evaluation
    results = evaluate_model(model_path, data_yaml)
    
    # 2. Generate evaluation report
    report = generate_evaluation_report(results, output_dir)
    
    # 3. Speed evaluation
    speed_metrics = calculate_speed_metrics(model_path, test_images_dir)
    
    # 4. Combine all metrics
    comprehensive_report = {
        "evaluation_report": report,
        "speed_metrics": speed_metrics
    }
    
    # Save comprehensive report
    with open(f"{output_dir}/comprehensive_evaluation.json", 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall mAP50: {report['performance_metrics']['mAP50']:.3f}")
    print(f"Overall mAP50-95: {report['performance_metrics']['mAP50-95']:.3f}")
    print(f"Precision: {report['performance_metrics']['precision']:.3f}")
    print(f"Recall: {report['performance_metrics']['recall']:.3f}")
    print(f"F1-Score: {report['performance_metrics']['f1_score']:.3f}")
    print(f"Average Inference Time: {speed_metrics.get('average_inference_time_ms', 0):.2f} ms")
    print(f"FPS: {speed_metrics.get('fps', 0):.2f}")
    print("="*50)
    
    return comprehensive_report

def main():
    parser = argparse.ArgumentParser(description="Smart Office Object Detection Evaluation")
    parser.add_argument("--model", required=True, help="Path to trained model weights")
    parser.add_argument("--data", default="smart_office_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--test-images", default="smart_office_dataset/test/images", help="Path to test images")
    parser.add_argument("--output", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--speed-test", action="store_true", help="Run speed evaluation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Run evaluation
    try:
        report = run_comprehensive_evaluation(
            args.model, 
            args.data, 
            args.test_images, 
            args.output
        )
        print(f"\nEvaluation completed successfully! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 