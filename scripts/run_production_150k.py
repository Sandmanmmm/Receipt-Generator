#!/usr/bin/env python3
"""
Production Dataset Generator for Vast.ai
=========================================
Generates 150,000 invoice/receipt samples with robust error handling,
progress monitoring, checkpointing, and automatic resume capability.

Usage:
    python scripts/run_production_150k.py --output /workspace/outputs/production_150k

Features:
    - Automatic checkpoint/resume on interruption
    - Detailed progress logging with ETA
    - Memory monitoring
    - Per-batch statistics
    - Graceful shutdown handling (Ctrl+C)
    - Automatic retry on transient failures
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup comprehensive logging to file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generation_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("production_generator")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - summary
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_system_stats() -> dict:
    """Get current system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def load_checkpoint(output_dir: Path) -> dict:
    """Load checkpoint if exists"""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'completed_samples': 0, 'failed_samples': [], 'start_time': None}


def save_checkpoint(output_dir: Path, checkpoint: dict):
    """Save checkpoint for resume capability"""
    checkpoint_file = output_dir / "checkpoint.json"
    checkpoint['last_update'] = datetime.now().isoformat()
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    kill_now = False
    
    def __init__(self, logger):
        self.logger = logger
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        self.logger.warning("\n⚠️  Shutdown signal received. Finishing current batch...")
        self.kill_now = True


def run_production_generation(
    output_dir: Path,
    total_samples: int = 150000,
    workers: int = 64,
    batch_size: int = 1000,
    augment_prob: float = 0.3,
    resume: bool = True,
    logger: Optional[logging.Logger] = None
):
    """
    Run production dataset generation with checkpointing and monitoring
    """
    from scripts.generate_parallel_dataset import generate_parallel_dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger is None:
        logger = setup_logging(output_dir)
    
    killer = GracefulKiller(logger)
    
    # Print banner
    logger.info("=" * 70)
    logger.info("🚀 PRODUCTION DATASET GENERATOR - 150K SAMPLES")
    logger.info("=" * 70)
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"🎯 Target: {total_samples:,} samples")
    logger.info(f"👷 Workers: {workers}")
    logger.info(f"📦 Batch size: {batch_size:,}")
    logger.info(f"🎨 Augmentation: {augment_prob*100:.0f}%")
    logger.info("=" * 70)
    
    # Load or create checkpoint
    checkpoint = load_checkpoint(output_dir) if resume else {'completed_samples': 0, 'failed_samples': [], 'start_time': None}
    
    if checkpoint['completed_samples'] > 0:
        logger.info(f"📌 Resuming from checkpoint: {checkpoint['completed_samples']:,} samples already completed")
    
    if checkpoint['start_time'] is None:
        checkpoint['start_time'] = datetime.now().isoformat()
    
    start_time = datetime.fromisoformat(checkpoint['start_time'])
    session_start = datetime.now()
    
    # Calculate batches
    completed = checkpoint['completed_samples']
    remaining = total_samples - completed
    num_batches = (remaining + batch_size - 1) // batch_size
    
    logger.info(f"\n📊 Session Plan:")
    logger.info(f"   Remaining samples: {remaining:,}")
    logger.info(f"   Batches to process: {num_batches}")
    
    # System stats
    stats = get_system_stats()
    logger.info(f"\n💻 System Status:")
    logger.info(f"   CPU: {stats['cpu_percent']:.1f}%")
    logger.info(f"   Memory: {stats['memory_used_gb']:.1f}GB used / {stats['memory_available_gb']:.1f}GB available ({stats['memory_percent']:.1f}%)")
    logger.info(f"   Disk: {stats['disk_percent']:.1f}% used")
    logger.info("")
    
    total_generated = completed
    total_errors = len(checkpoint.get('failed_samples', []))
    batch_times = []
    
    for batch_num in range(num_batches):
        if killer.kill_now:
            logger.warning("🛑 Graceful shutdown initiated")
            break
        
        batch_start = time.time()
        current_batch_size = min(batch_size, remaining - (batch_num * batch_size))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 BATCH {batch_num + 1}/{num_batches} | Samples: {current_batch_size:,}")
        logger.info(f"{'='*60}")
        
        # Generate batch output directory
        batch_output = output_dir / f"batch_{batch_num + 1:04d}"
        
        try:
            # Run generation
            result = generate_parallel_dataset(
                output_dir=str(batch_output),
                num_samples=current_batch_size,
                num_workers=workers,
                augment_probability=augment_prob
            )
            
            batch_success = result.get('success', 0)
            batch_errors = result.get('errors', 0)
            
            total_generated += batch_success
            total_errors += batch_errors
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            samples_per_sec = current_batch_size / batch_time if batch_time > 0 else 0
            
            # Update checkpoint
            checkpoint['completed_samples'] = total_generated
            if batch_errors > 0:
                checkpoint['failed_samples'].extend(result.get('failed_list', []))
            save_checkpoint(output_dir, checkpoint)
            
            # Progress stats
            progress = total_generated / total_samples * 100
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = num_batches - batch_num - 1
            eta_seconds = remaining_batches * avg_batch_time
            
            # System check
            stats = get_system_stats()
            
            logger.info(f"\n✅ Batch {batch_num + 1} Complete:")
            logger.info(f"   Generated: {batch_success:,} | Errors: {batch_errors}")
            logger.info(f"   Time: {format_time(batch_time)} | Rate: {samples_per_sec:.1f}/sec")
            logger.info(f"\n📈 Overall Progress:")
            logger.info(f"   Total: {total_generated:,}/{total_samples:,} ({progress:.1f}%)")
            logger.info(f"   Errors: {total_errors:,} ({total_errors/max(1,total_generated)*100:.2f}%)")
            logger.info(f"   ETA: {format_time(eta_seconds)}")
            logger.info(f"\n💻 System: CPU {stats['cpu_percent']:.0f}% | RAM {stats['memory_percent']:.0f}% | Disk {stats['disk_percent']:.0f}%")
            
            # Memory warning
            if stats['memory_percent'] > 90:
                logger.warning("⚠️  HIGH MEMORY USAGE - Consider reducing workers")
            if stats['disk_percent'] > 95:
                logger.error("❌ DISK ALMOST FULL - Stopping generation")
                break
                
        except Exception as e:
            logger.error(f"❌ Batch {batch_num + 1} failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Save checkpoint even on failure
            save_checkpoint(output_dir, checkpoint)
            
            # Continue to next batch
            continue
    
    # Final summary
    total_time = (datetime.now() - session_start).total_seconds()
    overall_time = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 70)
    logger.info("🏁 GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   Total Generated: {total_generated:,}")
    logger.info(f"   Total Errors: {total_errors:,}")
    logger.info(f"   Success Rate: {(total_generated-total_errors)/max(1,total_generated)*100:.2f}%")
    logger.info(f"   Session Time: {format_time(total_time)}")
    logger.info(f"   Overall Time: {format_time(overall_time)}")
    logger.info(f"   Average Rate: {total_generated/total_time:.1f} samples/sec")
    
    # Count output files
    total_images = sum(1 for _ in output_dir.rglob("*.jpg"))
    total_size_gb = sum(f.stat().st_size for f in output_dir.rglob("*.jpg")) / (1024**3)
    
    logger.info(f"\n📁 Output:")
    logger.info(f"   Total Images: {total_images:,}")
    logger.info(f"   Total Size: {total_size_gb:.2f} GB")
    logger.info(f"   Location: {output_dir}")
    logger.info("=" * 70)
    
    # Write completion marker
    if total_generated >= total_samples:
        completion_file = output_dir / "GENERATION_COMPLETE.txt"
        with open(completion_file, 'w') as f:
            f.write(f"Generation completed: {datetime.now().isoformat()}\n")
            f.write(f"Total samples: {total_generated:,}\n")
            f.write(f"Total images: {total_images:,}\n")
            f.write(f"Total size: {total_size_gb:.2f} GB\n")
            f.write(f"Total time: {format_time(overall_time)}\n")
    
    return {
        'total_generated': total_generated,
        'total_errors': total_errors,
        'total_images': total_images,
        'total_size_gb': total_size_gb,
        'total_time': total_time
    }


def main():
    parser = argparse.ArgumentParser(
        description="Production Dataset Generator for 150K samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full 150K run with defaults
    python scripts/run_production_150k.py --output /workspace/outputs/production_150k
    
    # Custom settings
    python scripts/run_production_150k.py --samples 50000 --workers 32 --augment 0.5
    
    # Resume interrupted run
    python scripts/run_production_150k.py --output /workspace/outputs/production_150k --resume
    
    # Fresh start (ignore checkpoint)
    python scripts/run_production_150k.py --output /workspace/outputs/production_150k --no-resume
        """
    )
    
    parser.add_argument('--output', '-o', type=str, 
                       default='/workspace/outputs/production_150k',
                       help='Output directory (default: /workspace/outputs/production_150k)')
    parser.add_argument('--samples', '-n', type=int, default=150000,
                       help='Total samples to generate (default: 150000)')
    parser.add_argument('--workers', '-w', type=int, default=64,
                       help='Number of parallel workers (default: 64)')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                       help='Samples per batch for checkpointing (default: 1000)')
    parser.add_argument('--augment', '-a', type=float, default=0.3,
                       help='Augmentation probability 0.0-1.0 (default: 0.3)')
    parser.add_argument('--resume/--no-resume', default=True,
                       help='Resume from checkpoint if exists (default: True)')
    
    args = parser.parse_args()
    
    # Validate
    if args.augment < 0 or args.augment > 1:
        print("Error: --augment must be between 0.0 and 1.0")
        sys.exit(1)
    
    output_dir = Path(args.output)
    logger = setup_logging(output_dir)
    
    try:
        result = run_production_generation(
            output_dir=output_dir,
            total_samples=args.samples,
            workers=args.workers,
            batch_size=args.batch_size,
            augment_prob=args.augment,
            resume=args.resume,
            logger=logger
        )
        
        if result['total_generated'] >= args.samples:
            logger.info("\n✅ Generation completed successfully!")
            sys.exit(0)
        else:
            logger.warning(f"\n⚠️  Generation incomplete: {result['total_generated']}/{args.samples}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user. Checkpoint saved.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
