#!/bin/bash
# ============================================================================
# InvoiceGen - One-Line Vast.ai Quick Start
# ============================================================================
# Copy and paste this entire script into your vast.ai terminal
# ============================================================================

# Quick setup (run as single command)
cd /workspace && \
apt-get update -qq && \
apt-get install -y --no-install-recommends wkhtmltopdf xvfb fonts-dejavu-core fonts-liberation git > /dev/null 2>&1 && \
git clone https://github.com/Sandmanmmm/Receipt-Generator.git InvoiceGen && \
cd InvoiceGen && \
pip install -q -r requirements.txt && \
pip install -q -e . && \
echo "Setup complete! Starting 150K generation in tmux..." && \
tmux new-session -d -s generate 'python scripts/generate_mixed_dataset.py --samples 150000 --output /workspace/outputs/production_150k 2>&1 | tee /workspace/generation.log' && \
echo "Generation started in background tmux session 'generate'" && \
echo "To monitor: tmux attach -t generate" && \
echo "To check progress: tail -f /workspace/generation.log"
