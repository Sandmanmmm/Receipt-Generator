"""
Vast.ai Integration Scripts
Run invoice generation at scale on vast.ai GPU instances
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict
import click


class VastAIManager:
    """Manage vast.ai instances for scaled generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize vast.ai manager
        
        Args:
            api_key: Vast.ai API key (or set VASTAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('VASTAI_API_KEY')
        if not self.api_key:
            raise ValueError("Vast.ai API key required. Set VASTAI_API_KEY env var.")
    
    def list_instances(self) -> List[Dict]:
        """List available GPU instances"""
        cmd = ['vastai', 'search', 'offers', '--raw']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            raise RuntimeError(f"Failed to list instances: {result.stderr}")
    
    def create_instance(self,
                       instance_id: int,
                       image: str = "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                       disk_space: int = 50) -> Dict:
        """
        Create a new vast.ai instance
        
        Args:
            instance_id: Instance ID from search results
            image: Docker image to use
            disk_space: Disk space in GB
            
        Returns:
            Instance details
        """
        cmd = [
            'vastai', 'create', 'instance',
            str(instance_id),
            '--image', image,
            '--disk', str(disk_space),
            '--raw'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            raise RuntimeError(f"Failed to create instance: {result.stderr}")
    
    def upload_code(self, instance_id: int, local_path: str = "."):
        """Upload code to instance"""
        # Get instance details
        cmd = ['vastai', 'show', 'instance', str(instance_id), '--raw']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get instance details: {result.stderr}")
        
        instance = json.loads(result.stdout)
        ssh_host = instance['ssh_host']
        ssh_port = instance['ssh_port']
        
        # Upload using rsync
        rsync_cmd = [
            'rsync', '-avz', '-e',
            f'ssh -p {ssh_port}',
            local_path + '/',
            f'root@{ssh_host}:/workspace/InvoiceGen/'
        ]
        
        result = subprocess.run(rsync_cmd)
        if result.returncode != 0:
            raise RuntimeError("Failed to upload code")
        
        click.echo(f"✓ Code uploaded to instance {instance_id}")
    
    def run_command(self, instance_id: int, command: str) -> str:
        """Execute command on instance"""
        cmd = ['vastai', 'ssh-url', str(instance_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get SSH URL: {result.stderr}")
        
        ssh_url = result.stdout.strip()
        
        # Execute command via SSH
        ssh_cmd = ['ssh', ssh_url, command]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        
        return result.stdout
    
    def destroy_instance(self, instance_id: int):
        """Destroy an instance"""
        cmd = ['vastai', 'destroy', 'instance', str(instance_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo(f"✓ Destroyed instance {instance_id}")
        else:
            raise RuntimeError(f"Failed to destroy instance: {result.stderr}")


class DistributedGenerator:
    """Run distributed invoice generation on multiple instances"""
    
    def __init__(self, manager: VastAIManager, config: Dict):
        """Initialize distributed generator"""
        self.manager = manager
        self.config = config
        self.instances = []
    
    def setup_instances(self, num_instances: int = 4):
        """Setup multiple vast.ai instances"""
        click.echo(f"Setting up {num_instances} instances...")
        
        # Find suitable instances
        available = self.manager.list_instances()
        suitable = [
            inst for inst in available
            if inst['gpu_name'] == self.config.get('gpu_name', 'RTX 4090')
            and inst['disk_space'] >= self.config.get('disk_space', 50)
        ]
        
        if len(suitable) < num_instances:
            raise RuntimeError(f"Only {len(suitable)} suitable instances available")
        
        # Create instances
        for i in range(num_instances):
            inst = suitable[i]
            created = self.manager.create_instance(
                instance_id=inst['id'],
                image=self.config['image'],
                disk_space=self.config['disk_space']
            )
            self.instances.append(created)
            click.echo(f"Created instance {i+1}/{num_instances}: {created['id']}")
        
        # Upload code to all instances
        for inst in self.instances:
            self.manager.upload_code(inst['id'])
    
    def generate_distributed(self,
                           total_samples: int,
                           output_dir: str = "data/vast_output"):
        """
        Run generation across multiple instances
        
        Args:
            total_samples: Total number of invoices to generate
            output_dir: Output directory for results
        """
        num_instances = len(self.instances)
        samples_per_instance = total_samples // num_instances
        
        click.echo(f"\nGenerating {total_samples} samples across {num_instances} instances")
        click.echo(f"Each instance will generate {samples_per_instance} samples\n")
        
        # Run generation on each instance
        for i, inst in enumerate(self.instances):
            start_idx = i * samples_per_instance
            end_idx = start_idx + samples_per_instance
            
            command = (
                f"cd /workspace/InvoiceGen && "
                f"python scripts/pipeline.py generate "
                f"--num-samples {samples_per_instance} "
                f"--seed {start_idx}"
            )
            
            click.echo(f"Starting generation on instance {inst['id']} ({start_idx}-{end_idx})...")
            self.manager.run_command(inst['id'], command)
        
        click.echo("\n✓ All instances started")
    
    def download_results(self, output_dir: str = "data/vast_output"):
        """Download results from all instances"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, inst in enumerate(self.instances):
            click.echo(f"Downloading from instance {inst['id']}...")
            
            # Get SSH details
            cmd = ['vastai', 'show', 'instance', str(inst['id']), '--raw']
            result = subprocess.run(cmd, capture_output=True, text=True)
            instance = json.loads(result.stdout)
            
            ssh_host = instance['ssh_host']
            ssh_port = instance['ssh_port']
            
            # Download using rsync
            rsync_cmd = [
                'rsync', '-avz', '-e',
                f'ssh -p {ssh_port}',
                f'root@{ssh_host}:/workspace/InvoiceGen/data/raw/',
                str(output_path / f'instance_{i}/')
            ]
            
            subprocess.run(rsync_cmd)
        
        click.echo("✓ Results downloaded")
    
    def cleanup(self):
        """Destroy all instances"""
        click.echo("\nCleaning up instances...")
        for inst in self.instances:
            self.manager.destroy_instance(inst['id'])
        
        click.echo("✓ Cleanup complete")


@click.group()
def cli():
    """Vast.ai integration for scaled invoice generation"""
    pass


@cli.command()
@click.option('--gpu', default='RTX 4090', help='GPU type to search for')
@click.option('--min-disk', default=50, help='Minimum disk space (GB)')
def search(gpu, min_disk):
    """Search for available instances"""
    manager = VastAIManager()
    instances = manager.list_instances()
    
    suitable = [
        inst for inst in instances
        if inst['gpu_name'] == gpu and inst['disk_space'] >= min_disk
    ]
    
    click.echo(f"\nFound {len(suitable)} suitable instances:\n")
    for inst in suitable[:10]:  # Show top 10
        click.echo(f"ID: {inst['id']}")
        click.echo(f"  GPU: {inst['gpu_name']}")
        click.echo(f"  Disk: {inst['disk_space']} GB")
        click.echo(f"  Price: ${inst['dph_total']:.3f}/hour")
        click.echo()


@cli.command()
@click.option('--num-instances', '-n', default=4, help='Number of instances')
@click.option('--total-samples', '-s', default=10000, help='Total samples to generate')
@click.option('--output-dir', '-o', default='data/vast_output', help='Output directory')
def generate(num_instances, total_samples, output_dir):
    """Run distributed generation"""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    vastai_config = config['vastai']
    
    # Initialize
    manager = VastAIManager()
    generator = DistributedGenerator(manager, vastai_config)
    
    try:
        # Setup instances
        generator.setup_instances(num_instances)
        
        # Generate
        generator.generate_distributed(total_samples, output_dir)
        
        # Wait for user to confirm completion
        click.echo("\nGeneration started on all instances.")
        click.echo("Monitor progress using: vastai show instances")
        input("\nPress Enter when generation is complete to download results...")
        
        # Download results
        generator.download_results(output_dir)
        
    finally:
        # Cleanup
        if click.confirm("\nDestroy instances?", default=True):
            generator.cleanup()


@cli.command()
@click.argument('instance_id', type=int)
def upload(instance_id):
    """Upload code to instance"""
    manager = VastAIManager()
    manager.upload_code(instance_id)


@cli.command()
@click.argument('instance_id', type=int)
@click.argument('command')
def exec(instance_id, command):
    """Execute command on instance"""
    manager = VastAIManager()
    output = manager.run_command(instance_id, command)
    click.echo(output)


if __name__ == '__main__':
    cli()
