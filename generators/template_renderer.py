"""
Template Renderer - Jinja2 Template Rendering
"""
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader


class TemplateRenderer:
    """Renders Jinja2 templates to HTML"""
    
    def __init__(self, templates_dir: str):
        """
        Initialize template renderer
        
        Args:
            templates_dir: Path to directory containing HTML templates
        """
        self.templates_dir = Path(templates_dir)
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
    
    def render(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Render template to HTML string
        
        Args:
            template_name: Name of template file (e.g., 'modern/invoice.html')
            data: Dictionary with template variables
            
        Returns:
            Rendered HTML string
        """
        template = self.env.get_template(template_name)
        return template.render(**data)
    
    def render_to_file(self, template_name: str, data: Dict[str, Any], 
                      output_path: str) -> str:
        """
        Render template and save to file
        
        Args:
            template_name: Name of template file
            data: Dictionary with template variables
            output_path: Path to save HTML file
            
        Returns:
            Rendered HTML string
        """
        html_content = self.render(template_name, data)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_content
    
    def list_templates(self, pattern: str = '*.html') -> list:
        """
        List available templates
        
        Args:
            pattern: Glob pattern for template files
            
        Returns:
            List of template names
        """
        templates = []
        for template_path in self.templates_dir.rglob(pattern):
            relative_path = template_path.relative_to(self.templates_dir)
            templates.append(str(relative_path).replace('\\', '/'))
        return templates


__all__ = ['TemplateRenderer']
