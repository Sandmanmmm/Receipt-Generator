"""
Visual Assets Generator
Generates logos, QR codes, and barcodes for invoices
"""
import qrcode
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import io
import base64
from pathlib import Path
from typing import Tuple, Optional, Union

class VisualAssetGenerator:
    """Generates visual assets for documents"""
    
    def __init__(self):
        self.logo_colors = [
            (44, 62, 80),    # Dark Blue
            (52, 152, 219),  # Blue
            (46, 204, 113),  # Green
            (231, 76, 60),   # Red
            (155, 89, 182),  # Purple
            (243, 156, 18),  # Orange
            (26, 188, 156),  # Teal
            (52, 73, 94),    # Navy
        ]
        
    def generate_qr_code(self, data: str, size: int = 200, color: str = "black") -> str:
        """
        Generate a QR code and return as base64 string
        
        Args:
            data: String data to encode
            size: Pixel size of the QR code
            color: Color of the QR code modules
            
        Returns:
            Base64 encoded PNG string
        """
        try:
            import qrcode
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=0,
            )
            qr.add_data(data)
            qr.make(fit=True)

            img = qr.make_image(fill_color=color, back_color="transparent")
            # Convert to standard PIL Image if it's a custom class
            if not isinstance(img, Image.Image):
                img = img.get_image()
                
            img = img.resize((size, size), Image.Resampling.NEAREST)
            
            return self._image_to_base64(img)
        except ImportError:
            # Fallback if qrcode not installed
            return self._generate_placeholder_qr(size, color)

    def _generate_placeholder_qr(self, size: int, color: str) -> str:
        """Generate a placeholder QR code (random noise)"""
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw random blocks
        block_size = size // 20
        for x in range(0, size, block_size):
            for y in range(0, size, block_size):
                if random.random() > 0.5:
                    draw.rectangle([x, y, x+block_size, y+block_size], fill=color)
                    
        # Draw corner markers
        marker_size = block_size * 4
        for x, y in [(0,0), (size-marker_size, 0), (0, size-marker_size)]:
            draw.rectangle([x, y, x+marker_size, y+marker_size], outline=color, width=block_size)
            draw.rectangle([x+block_size*2, y+block_size*2, x+marker_size-block_size*2, y+marker_size-block_size*2], fill=color)
            
        return self._image_to_base64(img)

    def generate_logo(self, company_name: str, width: int = 300, height: int = 100, 
                     color: Optional[Tuple[int, int, int]] = None) -> str:
        """
        Generate a simple company logo
        
        Args:
            company_name: Name of the company
            width: Image width
            height: Image height
            color: RGB tuple for primary color
            
        Returns:
            Base64 encoded PNG string
        """
        if color is None:
            color = random.choice(self.logo_colors)
            
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Logo style variants
        style = random.choice(['icon_left', 'icon_top', 'text_only', 'boxed'])
        
        # Get initials
        words = company_name.split()
        initials = "".join([w[0] for w in words[:2]]).upper()
        
        # Font setup (fallback to default if specific fonts missing)
        try:
            # Try to find a system font
            font_size = int(height * 0.6)
            font = ImageFont.truetype("arial.ttf", font_size)
            small_font = ImageFont.truetype("arial.ttf", int(font_size * 0.4))
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        if style == 'icon_left':
            # Draw icon (circle/square with initials)
            icon_size = int(height * 0.8)
            margin = (height - icon_size) // 2
            
            # Icon shape
            shape_type = random.choice(['circle', 'square', 'rounded'])
            if shape_type == 'circle':
                draw.ellipse([margin, margin, margin+icon_size, margin+icon_size], fill=color)
            elif shape_type == 'square':
                draw.rectangle([margin, margin, margin+icon_size, margin+icon_size], fill=color)
            else:
                # Rounded rect approximation
                draw.rectangle([margin, margin, margin+icon_size, margin+icon_size], fill=color)
            
            # Initials in icon
            # Center text in icon (approximate)
            draw.text((margin + icon_size//3, margin + icon_size//4), initials, fill='white', font=small_font)
            
            # Company name
            draw.text((margin + icon_size + 20, margin + 10), company_name, fill=color, font=small_font)
            
        elif style == 'icon_top':
            # Icon above text
            icon_size = int(height * 0.5)
            icon_x = (width - icon_size) // 2
            
            draw.rectangle([icon_x, 0, icon_x+icon_size, icon_size], fill=color)
            draw.text((icon_x + icon_size//3, icon_size//4), initials, fill='white', font=small_font)
            
            # Text below
            text_width = draw.textlength(company_name, font=small_font)
            text_x = (width - text_width) // 2
            draw.text((text_x, icon_size + 5), company_name, fill='black', font=small_font)
            
        elif style == 'boxed':
            # Text inside a box
            draw.rectangle([0, 0, width-1, height-1], outline=color, width=3)
            draw.rectangle([0, 0, 20, height], fill=color)
            
            draw.text((30, height//4), company_name.upper(), fill='black', font=small_font)
            
        else: # text_only
            # Just stylized text
            draw.text((0, height//4), company_name, fill=color, font=font)
            
        return self._image_to_base64(img)

    def generate_barcode(self, value: str, width: int = 300, height: int = 80) -> str:
        """
        Generate a barcode-like image (Code 128 style approximation)
        
        Args:
            value: String value to encode (displayed below barcode)
            width: Image width
            height: Image height
            
        Returns:
            Base64 encoded PNG string
        """
        # Create white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw random vertical lines to simulate barcode
        # Leave margins
        margin_x = 20
        margin_y = 10
        bar_height = height - 30 # Leave space for text
        
        current_x = margin_x
        while current_x < width - margin_x:
            # Random bar width
            bar_w = random.choice([1, 2, 3, 4])
            # Random gap
            gap_w = random.choice([1, 2, 3])
            
            if current_x + bar_w > width - margin_x:
                break
                
            draw.rectangle([current_x, margin_y, current_x + bar_w, margin_y + bar_height], fill='black')
            current_x += bar_w + gap_w
            
        # Draw text value below
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
            
        # Center text
        text_bbox = draw.textbbox((0, 0), value, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = (width - text_w) // 2
        draw.text((text_x, height - 15), value, fill='black', font=font)
        
        return self._image_to_base64(img)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

if __name__ == "__main__":
    # Test generation
    gen = VisualAssetGenerator()
    print("Generating test assets...")
    logo = gen.generate_logo("Acme Corp")
    qr = gen.generate_qr_code("INV-12345")
    barcode = gen.generate_barcode("INV-12345")
    print(f"Logo length: {len(logo)}")
    print(f"QR length: {len(qr)}")
    print(f"Barcode length: {len(barcode)}")
