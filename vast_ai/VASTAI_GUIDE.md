# InvoiceGen - Vast.ai Dataset Generation Guide

## Overview

This guide explains how to use Vast.ai with an RTX 5090 GPU to generate the 150,000 document training dataset for LayoutLMv3.

## Why Vast.ai + RTX 5090?

| Aspect | Benefit |
|--------|---------|
| **Speed** | ~4x faster than local generation due to faster CPU/IO |
| **Reliability** | Dedicated instance won't be interrupted |
| **Cost** | RTX 5090 on Vast.ai: ~$0.50-1.00/hour |
| **Disk** | 50-100GB SSD included |

**Note**: For dataset *generation*, GPU is not heavily used (wkhtmltoimage is CPU-bound). However, the faster CPUs and SSDs on GPU instances significantly speed up generation. GPU will be critical for subsequent model training.

## Quick Start

### 1. Rent a Vast.ai Instance

1. Go to [vast.ai/console/create](https://cloud.vast.ai/console/create/)
2. Search for instances with:
   - **GPU**: RTX 5090 (or RTX 4090 as alternative)
   - **Disk**: 50GB+ (100GB recommended)
   - **Image**: `nvidia/cuda:12.4-devel-ubuntu22.04` or `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
   - **vCPU**: 8+ cores recommended
3. Click "RENT" on a suitable instance

### 2. Connect to Instance

```bash
# SSH into your instance (vast.ai provides the command)
ssh -p <port> root@<instance-ip> -L 8080:localhost:8080
```

### 3. Run Setup Script

```bash
# Download and run the setup script
cd /workspace
git clone https://github.com/Sandmanmmm/Receipt-Generator.git InvoiceGen
cd InvoiceGen
chmod +x vast_ai/*.sh
./vast_ai/setup_vastai.sh
```

### 4. Start Generation (in tmux)

```bash
# Start a tmux session (survives disconnects)
tmux new -s generate

# Run the generation
./vast_ai/run_generation.sh 150000 /workspace/outputs/production_150k

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t generate
```

## Estimated Time & Cost

| Metric | Estimate |
|--------|----------|
| **Documents** | 150,000 |
| **Pages** | ~282,000 (1.88 pages/doc avg) |
| **Disk Space** | ~15 GB |
| **Generation Time** | 25-35 hours |
| **Vast.ai Cost** | ~$15-25 (at $0.50-0.70/hr) |

## Monitoring Progress

### Check Generation Status

```bash
# Reattach to tmux
tmux attach -t generate

# Or check the log file
tail -f /workspace/logs/generation_*.log

# Check disk usage
du -sh /workspace/outputs/production_150k/

# Count generated images
ls /workspace/outputs/production_150k/images/*.png | wc -l
```

### GPU/CPU Monitoring

```bash
# GPU utilization (will be low during generation)
nvidia-smi -l 5

# CPU and memory
htop
```

## Downloading Results

### Option 1: Direct SCP (Fast)

```bash
# From your local machine
scp -P <port> -r root@<instance-ip>:/workspace/outputs/production_150k ./data/
```

### Option 2: Compress First (Better for slow connections)

```bash
# On the vast.ai instance
cd /workspace/outputs
tar -czvf production_150k.tar.gz production_150k/

# Then download the archive
scp -P <port> root@<instance-ip>:/workspace/outputs/production_150k.tar.gz ./
```

### Option 3: Upload to Cloud Storage

```bash
# On vast.ai instance - upload to Google Cloud
pip install google-cloud-storage
gsutil -m cp -r /workspace/outputs/production_150k gs://your-bucket/

# Or AWS S3
pip install awscli
aws s3 sync /workspace/outputs/production_150k s3://your-bucket/production_150k/
```

## Troubleshooting

### wkhtmltoimage Not Found

```bash
# Reinstall
apt-get update && apt-get install -y wkhtmltopdf
which wkhtmltoimage  # Should show /usr/bin/wkhtmltoimage
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up intermediate files
rm -rf /tmp/*.html /tmp/*.png

# If needed, generate in smaller batches
python scripts/generate_mixed_dataset.py --samples 50000 --output /workspace/outputs/batch1
python scripts/generate_mixed_dataset.py --samples 50000 --output /workspace/outputs/batch2
python scripts/generate_mixed_dataset.py --samples 50000 --output /workspace/outputs/batch3
```

### Generation Interrupted

The generation script saves progress. If interrupted:

1. Check how many images were generated
2. Delete incomplete metadata files
3. Restart with remaining sample count

```bash
# Count existing images
EXISTING=$(ls /workspace/outputs/production_150k/images/*.png 2>/dev/null | wc -l)
echo "Existing: $EXISTING images"

# Continue with remaining (if using batch script)
REMAINING=$((282000 - EXISTING))
echo "Remaining: ~$REMAINING images to generate"
```

## File Structure After Generation

```
/workspace/outputs/production_150k/
├── images/                    # ~282,000 PNG files (~15 GB)
│   ├── receipt_00000_page1.png
│   ├── receipt_00001_page1.png
│   ├── invoice_00000_page1.png
│   ├── invoice_00000_page2.png
│   └── ...
├── metadata/                  # ~150,000 JSON files (~500 MB)
│   ├── receipt_00000.json
│   ├── invoice_00000.json
│   └── ...
└── COMPLETE.txt              # Completion marker with stats
```

## Next Steps After Generation

1. **Download** the dataset to your local machine or training server
2. **Run OCR annotation** pipeline to create labeled training data
3. **Split dataset** into train/val/test (80/10/10)
4. **Train LayoutLMv3** model (separate vast.ai session with GPU)

## Cost Optimization Tips

1. **Use spot instances** if available (cheaper but can be interrupted)
2. **Pause billing** immediately after generation completes
3. **Use disk persistence** to save progress between sessions
4. **Compress before download** to reduce transfer time

## Instance Recommendations

| Provider | GPU | vCPU | RAM | Disk | Price/hr |
|----------|-----|------|-----|------|----------|
| Vast.ai | RTX 5090 | 16+ | 32GB+ | 100GB SSD | ~$0.70-1.00 |
| Vast.ai | RTX 4090 | 12+ | 24GB+ | 100GB SSD | ~$0.40-0.60 |
| Vast.ai | A100 40GB | 16+ | 64GB+ | 200GB SSD | ~$1.50-2.00 |

**Recommended**: RTX 4090 or 5090 - best price/performance for generation (CPU-bound task)

---

## Support

For issues:
1. Check the log files in `/workspace/logs/`
2. Verify wkhtmltoimage installation
3. Check disk space with `df -h`
4. Open an issue on the GitHub repository
