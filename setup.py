"""
Setup and Installation Helper
Checks dependencies and sets up the environment
"""
import subprocess
import sys
import os
from pathlib import Path
import click


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        click.echo("❌ Python 3.9+ required", err=True)
        return False
    
    click.echo(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_command(command: str, name: str) -> bool:
    """Check if a command is available"""
    try:
        subprocess.run(
            [command, '--version'],
            capture_output=True,
            check=True
        )
        click.echo(f"✓ {name} installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.echo(f"⚠ {name} not found")
        return False


def check_gpu():
    """Check for GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            click.echo(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            click.echo("⚠ No GPU detected (will use CPU)")
            return False
    except ImportError:
        click.echo("⚠ PyTorch not installed yet")
        return False


def install_requirements():
    """Install Python requirements"""
    click.echo("\nInstalling Python dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            check=True
        )
        click.echo("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        click.echo("❌ Failed to install dependencies", err=True)
        return False


def create_env_file():
    """Create .env file from template"""
    if Path('.env').exists():
        click.echo("✓ .env file already exists")
        return True
    
    if Path('.env.example').exists():
        import shutil
        shutil.copy('.env.example', '.env')
        click.echo("✓ Created .env file (please edit with your settings)")
        return True
    
    return False


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'data/layoutlmv3/train',
        'data/layoutlmv3/val',
        'data/layoutlmv3/test',
        'models',
        'outputs',
        'logs',
        'evaluation'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    click.echo("✓ Created directory structure")


@click.command()
@click.option('--install-deps/--no-install-deps', default=True, help='Install Python dependencies')
@click.option('--check-only', is_flag=True, help='Only check environment, do not install')
def setup(install_deps, check_only):
    """Setup InvoiceGen environment"""
    
    click.echo("="*60)
    click.echo("INVOICEGEN SETUP")
    click.echo("="*60)
    
    # Check Python version
    click.echo("\n1. Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Check system dependencies
    click.echo("\n2. Checking system dependencies...")
    has_wkhtmltopdf = check_command('wkhtmltopdf', 'wkhtmltopdf')
    has_poppler = check_command('pdfinfo', 'Poppler')
    
    if not has_wkhtmltopdf:
        click.echo("   Install from: https://wkhtmltopdf.org/downloads.html")
    
    if not has_poppler:
        click.echo("   Windows: https://github.com/oschwartz10612/poppler-windows/releases/")
        click.echo("   Linux: sudo apt-get install poppler-utils")
        click.echo("   Mac: brew install poppler")
    
    # Check GPU
    click.echo("\n3. Checking GPU...")
    has_gpu = check_gpu()
    
    if check_only:
        click.echo("\n" + "="*60)
        click.echo("Check complete!")
        return
    
    # Install dependencies
    if install_deps:
        click.echo("\n4. Installing Python dependencies...")
        if not install_requirements():
            sys.exit(1)
    
    # Setup directories
    click.echo("\n5. Setting up directories...")
    setup_directories()
    
    # Create .env file
    click.echo("\n6. Creating environment file...")
    create_env_file()
    
    # Final summary
    click.echo("\n" + "="*60)
    click.echo("SETUP COMPLETE!")
    click.echo("="*60)
    
    click.echo("\nNext steps:")
    click.echo("  1. Edit .env file with your API keys (optional)")
    click.echo("  2. Run quick start: python scripts/quickstart.py")
    click.echo("  3. Generate dataset: python scripts/pipeline.py generate -n 100")
    click.echo("  4. Train model: python scripts/pipeline.py pipeline -n 100")
    click.echo("  5. See README.md for full documentation")
    
    if not has_wkhtmltopdf and not has_poppler:
        click.echo("\n⚠ Warning: Install wkhtmltopdf and Poppler for full functionality")
    
    if not has_gpu:
        click.echo("\n⚠ Note: Training will be slower on CPU")


if __name__ == '__main__':
    setup()
