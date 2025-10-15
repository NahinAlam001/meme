"""
Main entry point for training RGDORA on BHM dataset.
Supports both Task 1 (binary) and Task 2 (multi-class).
"""

import argparse
from config import RGDORAConfig
from trainer import RGDORATrainer


def train_task(task_name: str):
    """Train model for specified task"""
    print(f"\n{'='*80}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*80}\n")
    
    # Create task-specific configuration
    if task_name == "task1":
        config = RGDORAConfig(
            task="task1",
            batch_size=8,
            epochs=20,
            patience=5,
            learning_rate=2e-5,
            loss_type="focal",
            use_balanced_sampler=True,
            sampler_alpha=0.7,
            monitor_metric="val_macro_f1"
        )
    elif task_name == "task2":
        config = RGDORAConfig(
            task="task2",
            batch_size=8,
            epochs=25,
            patience=7,
            learning_rate=1.5e-5,
            loss_type="class_balanced",
            use_balanced_sampler=True,
            sampler_alpha=0.9,
            monitor_metric="val_macro_f1",
            focal_alpha=0.5,
            cb_beta=0.9999
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Initialize trainer
    trainer = RGDORATrainer(config)
    
    # Train
    trainer.train(resume=True)
    
    # Evaluate
    test_metrics = trainer.evaluate(split="test")
    
    print(f"\n✅ {task_name.upper()} Training Completed!")
    
    return test_metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train RGDORA for Bengali Hateful Meme Detection")
    parser.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2", "both"],
        default="task1",
        help="Which task to train: task1 (binary), task2 (multi-class), or both"
    )
    
    args = parser.parse_args()
    
    if args.task == "both":
        # Train both tasks sequentially
        print("\n" + "🎯"*40)
        print("TRAINING BOTH TASKS")
        print("🎯"*40 + "\n")
        
        # Task 1
        task1_metrics = train_task("task1")
        
        # Task 2
        task2_metrics = train_task("task2")
        
        # Summary
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print("\nTask 1 (Binary - Hate vs Non-Hate):")
        print(f"  Accuracy: {task1_metrics.get('test_accuracy', 0):.4f}")
        print(f"  Macro F1: {task1_metrics.get('test_macro_f1', 0):.4f}")
        print(f"  Balanced Accuracy: {task1_metrics.get('test_balanced_accuracy', 0):.4f}")
        print(f"  MCC: {task1_metrics.get('test_mcc', 0):.4f}")
        
        print("\nTask 2 (Multi-class - TI, TC, TO, TS):")
        print(f"  Accuracy: {task2_metrics.get('test_accuracy', 0):.4f}")
        print(f"  Macro F1: {task2_metrics.get('test_macro_f1', 0):.4f}")
        print(f"  Balanced Accuracy: {task2_metrics.get('test_balanced_accuracy', 0):.4f}")
        print(f"  MCC: {task2_metrics.get('test_mcc', 0):.4f}")
        print("="*80)
        
    else:
        # Train single task
        train_task(args.task)


if __name__ == "__main__":
    main()
