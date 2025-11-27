"""
Test Docker Deployment
"""
import pytest
import subprocess
import time
import requests
from pathlib import Path


@pytest.mark.docker
class TestDockerBuild:
    """Test Docker image building"""
    
    def test_dockerfile_exists(self):
        assert Path("Dockerfile").exists()
        assert Path("docker-compose.yml").exists()
        assert Path(".dockerignore").exists()
    
    @pytest.mark.slow
    def test_build_image(self):
        """Test building Docker image"""
        result = subprocess.run(
            ["docker", "build", "-t", "invoicegen:test", "."],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Build failed: {result.stderr}"
    
    @pytest.mark.slow
    def test_docker_compose_config(self):
        """Test docker-compose configuration"""
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Config invalid: {result.stderr}"


@pytest.mark.docker
@pytest.mark.slow
class TestDockerAPI:
    """Test Docker API service"""
    
    @pytest.fixture(scope="class")
    def api_service(self):
        """Start API service"""
        # Start service
        subprocess.run(
            ["docker-compose", "up", "-d", "invoicegen-api"],
            check=True
        )
        
        # Wait for service to be ready
        time.sleep(10)
        
        yield "http://localhost:8000"
        
        # Cleanup
        subprocess.run(
            ["docker-compose", "down"],
            check=True
        )
    
    def test_health_endpoint(self, api_service):
        """Test health check endpoint"""
        response = requests.get(f"{api_service}/health", timeout=10)
        assert response.status_code == 200
    
    def test_api_accessibility(self, api_service):
        """Test API is accessible"""
        try:
            response = requests.get(f"{api_service}/docs", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.fail("API not accessible")


@pytest.mark.docker
class TestDockerVolumes:
    """Test Docker volume mounts"""
    
    def test_volume_mounts_in_compose(self):
        """Check volume mounts are properly configured"""
        import yaml
        
        with open("docker-compose.yml", 'r') as f:
            config = yaml.safe_load(f)
        
        api_service = config['services']['invoicegen-api']
        
        assert 'volumes' in api_service
        volumes = api_service['volumes']
        
        # Check expected mounts
        volume_paths = [v.split(':')[1] for v in volumes]
        assert '/app/models' in volume_paths
        assert '/app/data' in volume_paths


@pytest.mark.docker
class TestDockerEnvironment:
    """Test Docker environment configuration"""
    
    def test_required_env_vars(self):
        """Check required environment variables"""
        import yaml
        
        with open("docker-compose.yml", 'r') as f:
            config = yaml.safe_load(f)
        
        api_service = config['services']['invoicegen-api']
        
        if 'environment' in api_service:
            env = api_service['environment']
            # Check some expected variables exist
            assert any('MODEL_PATH' in var for var in env)


@pytest.mark.docker
class TestDockerNetworking:
    """Test Docker networking"""
    
    def test_network_configuration(self):
        """Check network is properly configured"""
        import yaml
        
        with open("docker-compose.yml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'networks' in config
        assert 'invoicegen-network' in config['networks']
        
        # Check services use the network
        for service in config['services'].values():
            if 'networks' in service:
                assert 'invoicegen-network' in service['networks']


@pytest.mark.docker
class TestDockerIgnore:
    """Test .dockerignore patterns"""
    
    def test_dockerignore_patterns(self):
        """Verify .dockerignore contains important patterns"""
        with open(".dockerignore", 'r') as f:
            content = f.read()
        
        # Check important patterns are ignored
        assert '__pycache__' in content
        assert '.git' in content or '*.git' in content
        assert 'data/' in content or 'data/*' in content
        assert 'models/' in content or 'models/*' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'docker'])
