#!/bin/bash
# ============================================================================
# InvoiceGen - Vast.ai Remote Dataset Generation Setup Script
# ============================================================================
# This script sets up the environment on a vast.ai GPU instance
# Optimized for RTX 5090 with 32GB VRAM
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "InvoiceGen - Vast.ai Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/Sandmanmmm/Receipt-Generator.git"
WORKSPACE="/workspace"
OUTPUT_DIR="/workspace/outputs/production_150k"

# Step 1: System Updates
echo -e "${YELLOW}[1/7] Updating system packages...${NC}"
apt-get update -qq
apt-get install -y --no-install-recommends \
    wkhtmltopdf \
    xvfb \
    libxrender1 \
    libfontconfig1 \
    libx11-dev \
    libjpeg-turbo8 \
    libpng16-16 \
    poppler-utils \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-freefont-ttf \
    fontconfig \
    tesseract-ocr \
    tesseract-ocr-eng \
    git \
    wget \
    htop \
    tmux \
    > /dev/null 2>&1

fc-cache -fv > /dev/null 2>&1
echo -e "${GREEN}✓ System packages installed${NC}"

# Step 2: Clone or Update Repository
echo -e "${YELLOW}[2/7] Setting up repository...${NC}"
cd $WORKSPACE
if [ -d "InvoiceGen" ]; then
    echo "Repository exists, pulling latest..."
    cd InvoiceGen
    git pull
else
    echo "Cloning repository..."
    git clone $REPO_URL InvoiceGen
    cd InvoiceGen
fi
echo -e "${GREEN}✓ Repository ready${NC}"

# Step 3: Python Environment
echo -e "${YELLOW}[3/7] Setting up Python environment...${NC}"
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
pip install -e . -q
echo -e "${GREEN}✓ Python packages installed${NC}"

# Step 4: Verify wkhtmltoimage
echo -e "${YELLOW}[4/7] Verifying wkhtmltoimage...${NC}"
if command -v wkhtmltoimage &> /dev/null; then
    WKHTML_VERSION=$(wkhtmltoimage --version 2>&1 | head -1)
    echo -e "${GREEN}✓ wkhtmltoimage found: $WKHTML_VERSION${NC}"
else
    echo -e "${RED}✗ wkhtmltoimage not found!${NC}"
    exit 1
fi

# Step 5: Verify GPU
echo -e "${YELLOW}[5/7] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU detected: $GPU_INFO${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected (will use CPU for OCR)${NC}"
fi

# Step 6: Create output directory
echo -e "${YELLOW}[6/7] Creating output directory...${NC}"
mkdir -p $OUTPUT_DIR
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"

# Step 7: Quick validation test
echo -e "${YELLOW}[7/7] Running validation test (20 samples)...${NC}"
cd $WORKSPACE/InvoiceGen
python scripts/generate_parallel_dataset.py --samples 20 --workers 8 --augment 0.3 --output /workspace/outputs/test_validation

# Check result
if [ $? -eq 0 ]; then
    TEST_COUNT=$(ls /workspace/outputs/test_validation/images/*.jpg 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Validation passed: $TEST_COUNT images generated${NC}"
    rm -rf /workspace/outputs/test_validation  # Cleanup test
else
    echo -e "${RED}✗ Validation failed!${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To start 150K production generation:"
echo ""
echo "  cd $WORKSPACE/InvoiceGen"
echo "  tmux new-session -d -s generate"
echo "  tmux send-keys -t generate 'python scripts/run_production_150k.py \\"
echo "      --output $OUTPUT_DIR \\"
echo "      --samples 150000 \\"
echo "      --workers 64 \\"
echo "      --batch-size 1000 \\"
echo "      --augment 0.3 2>&1 | tee /workspace/generation.log' Enter"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t generate          # Live output (Ctrl+B, D to detach)"
echo "  tail -f /workspace/generation.log  # Log file"
echo "  htop                              # System resources"
echo ""
echo "Features:"
echo "  - Automatic checkpointing every 1000 samples"
echo "  - Resume from interruption with --resume"
echo "  - Graceful shutdown on Ctrl+C"
echo ""
echo "Estimated time: ~8-10 hours (64 workers)"
echo "Estimated disk usage: ~25-35 GB"
echo ""
