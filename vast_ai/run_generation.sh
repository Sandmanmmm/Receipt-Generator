#!/bin/bash
# ============================================================================
# InvoiceGen - Production 150K Dataset Generation Script
# ============================================================================
# Run this in a tmux session on vast.ai for long-running generation
# Usage: ./run_generation.sh [samples] [output_dir]
# ============================================================================

set -e

# Configuration (can be overridden by command line args)
SAMPLES=${1:-150000}
OUTPUT_DIR=${2:-/workspace/outputs/production_150k}
BATCH_SIZE=10000  # Generate in batches for fault tolerance

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "InvoiceGen - Production Generation"
echo "=========================================="
echo "Samples: $SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cd /workspace/InvoiceGen

# Create timestamped log file
LOG_FILE="/workspace/logs/generation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /workspace/logs

echo -e "${YELLOW}Starting generation at $(date)${NC}"
echo "Log file: $LOG_FILE"

# Run with nohup to survive disconnects, and tee to log
python scripts/generate_mixed_dataset.py \
    --samples $SAMPLES \
    --output $OUTPUT_DIR \
    2>&1 | tee $LOG_FILE

# Check completion
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Generation Complete!"
    echo "==========================================${NC}"
    
    # Stats
    IMG_COUNT=$(ls $OUTPUT_DIR/images/*.png 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
    
    echo "Images generated: $IMG_COUNT"
    echo "Total size: $TOTAL_SIZE"
    echo "Completed at: $(date)"
    
    # Create completion marker
    echo "$(date): Generation complete - $IMG_COUNT images" > $OUTPUT_DIR/COMPLETE.txt
else
    echo "Generation failed! Check $LOG_FILE for details."
    exit 1
fi
