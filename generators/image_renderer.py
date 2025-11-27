"""
Image Renderer - PDF to Image Conversion
"""
from pathlib import Path
from typing import Optional, Literal


class ImageRenderer:
    """Converts PDF to images"""
    
    def __init__(self, dpi: int = 150):
        """
        Initialize image renderer
        
        Args:
            dpi: Resolution in DPI (default: 150)
        """
        self.dpi = dpi
    
    def pdf_to_image(self, pdf_path: str, image_path: str, 
                    format: Literal['PNG', 'JPEG'] = 'PNG',
                    page: int = 1) -> bool:
        """
        Convert PDF page to image
        
        Args:
            pdf_path: Path to PDF file
            image_path: Output image path
            format: Image format ('PNG' or 'JPEG')
            page: Page number to convert (1-indexed)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page,
                last_page=page
            )
            
            if images:
                output_file = Path(image_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save image
                images[0].save(image_path, format)
                return True
            
            return False
        
        except ImportError:
            print("pdf2image not installed. Install with: pip install pdf2image")
            print("Also requires poppler:")
            print("  Windows: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("  Linux: sudo apt-get install poppler-utils")
            print("  Mac: brew install poppler")
            return False
        except Exception as e:
            print(f"PDF to image conversion error: {e}")
            return False
    
    def pdf_to_images(self, pdf_path: str, output_dir: str,
                     format: Literal['PNG', 'JPEG'] = 'PNG',
                     prefix: str = 'page') -> list:
        """
        Convert all PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for images
            format: Image format ('PNG' or 'JPEG')
            prefix: Filename prefix for images
            
        Returns:
            List of paths to generated images
        """
        try:
            from pdf2image import convert_from_path
            
            # Convert all pages
            images = convert_from_path(pdf_path, dpi=self.dpi)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_paths = []
            ext = format.lower()
            
            for idx, image in enumerate(images, 1):
                image_file = output_path / f"{prefix}_{idx:03d}.{ext}"
                image.save(image_file, format)
                image_paths.append(str(image_file))
            
            return image_paths
        
        except ImportError:
            print("pdf2image not installed. Install with: pip install pdf2image")
            return []
        except Exception as e:
            print(f"PDF to images conversion error: {e}")
            return []
    
    def resize_image(self, input_path: str, output_path: str,
                    width: Optional[int] = None, height: Optional[int] = None,
                    maintain_aspect: bool = True) -> bool:
        """
        Resize image
        
        Args:
            input_path: Path to input image
            output_path: Path to output image
            width: Target width in pixels (None to auto-calculate)
            height: Target height in pixels (None to auto-calculate)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from PIL import Image
            
            img = Image.open(input_path)
            
            if width is None and height is None:
                # No resize needed
                img.save(output_path)
                return True
            
            if maintain_aspect:
                # Calculate size maintaining aspect ratio
                if width and not height:
                    aspect = img.height / img.width
                    height = int(width * aspect)
                elif height and not width:
                    aspect = img.width / img.height
                    width = int(height * aspect)
                else:
                    # Both specified, calculate to fit within bounds
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                    img.save(output_path)
                    return True
            
            # Resize image
            resized = img.resize((width, height), Image.Resampling.LANCZOS)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            resized.save(output_path)
            return True
        
        except ImportError:
            print("PIL/Pillow not installed. Install with: pip install Pillow")
            return False
        except Exception as e:
            print(f"Image resize error: {e}")
            return False


__all__ = ['ImageRenderer']
