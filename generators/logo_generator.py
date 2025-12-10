"""
Enhanced Logo Generator
Generates high-quality, deterministic logos for synthetic invoices using PIL
Category-aware designs with consistent branding per company
"""
import hashlib
import random
from typing import Tuple, Optional, Dict, List
from PIL import Image, ImageDraw, ImageFont
import io
import base64


class LogoGenerator:
    """Enhanced logo generator with category-aware, deterministic designs"""
    
    # Category-specific color palettes (primary, secondary, accent)
    CATEGORY_PALETTES = {
        'fashion': {
            'luxury': [(0, 0, 0), (212, 175, 55), (255, 255, 255)],  # Black, Gold, White
            'modern': [(51, 51, 51), (189, 189, 189), (255, 255, 255)],  # Dark Gray, Silver, White
            'casual': [(44, 62, 80), (52, 152, 219), (255, 255, 255)]  # Navy, Blue, White
        },
        'electronics': {
            'tech': [(52, 152, 219), (44, 62, 80), (236, 240, 241)],  # Blue, Navy, Light
            'modern': [(41, 128, 185), (52, 73, 94), (149, 165, 166)],  # Blue, Dark, Gray
            'minimal': [(93, 109, 126), (255, 255, 255), (52, 152, 219)]  # Gray, White, Blue
        },
        'beauty': {
            'elegant': [(219, 112, 147), (255, 192, 203), (255, 255, 255)],  # Pink, Light Pink, White
            'luxury': [(186, 85, 211), (218, 112, 214), (255, 255, 255)],  # Purple, Orchid, White
            'natural': [(210, 180, 140), (255, 228, 196), (255, 255, 255)]  # Tan, Bisque, White
        },
        'accessories': {
            'modern': [(52, 73, 94), (149, 165, 166), (255, 255, 255)],
            'luxury': [(139, 69, 19), (210, 180, 140), (255, 255, 255)]
        },
        'jewelry': {
            'luxury': [(212, 175, 55), (184, 134, 11), (255, 255, 255)],  # Gold tones
            'elegant': [(192, 192, 192), (169, 169, 169), (255, 255, 255)]  # Silver tones
        },
        'home_garden': {
            'natural': [(34, 139, 34), (107, 142, 35), (255, 255, 255)],  # Green tones
            'modern': [(70, 130, 180), (95, 158, 160), (255, 255, 255)]
        },
        'sports_fitness': {
            'energy': [(255, 69, 0), (255, 140, 0), (255, 255, 255)],  # Orange tones
            'modern': [(30, 144, 255), (0, 191, 255), (255, 255, 255)]  # Blue tones
        },
        'food_beverage': {
            'warm': [(255, 99, 71), (255, 165, 0), (255, 255, 255)],  # Red, Orange
            'natural': [(139, 69, 19), (210, 105, 30), (255, 255, 255)]  # Brown tones
        },
        'health_wellness': {
            'natural': [(102, 205, 170), (32, 178, 170), (255, 255, 255)],  # Aqua, Teal
            'modern': [(135, 206, 250), (70, 130, 180), (255, 255, 255)]
        },
        'pet_supplies': {
            'playful': [(255, 165, 0), (255, 215, 0), (255, 255, 255)],  # Orange, Gold
            'natural': [(139, 69, 19), (160, 82, 45), (255, 255, 255)]
        },
        'books_media': {
            'classic': [(139, 69, 19), (160, 82, 45), (255, 255, 255)],
            'modern': [(70, 130, 180), (100, 149, 237), (255, 255, 255)]
        },
        'toys_games': {
            'playful': [(255, 20, 147), (255, 105, 180), (255, 255, 255)],
            'bright': [(255, 69, 0), (255, 215, 0), (255, 255, 255)]
        },
        'default': {
            'neutral': [(44, 62, 80), (52, 152, 219), (255, 255, 255)]
        }
    }
    
    # Font preferences by category
    FONT_PREFERENCES = {
        'fashion': ['georgia.ttf', 'times.ttf'],  # Serif for elegance
        'electronics': ['arial.ttf', 'helvetica.ttf'],  # Sans-serif modern
        'beauty': ['georgia.ttf', 'palatino.ttf'],
        'luxury': ['georgia.ttf', 'times.ttf'],
        'casual': ['arial.ttf', 'verdana.ttf'],
        'default': ['arial.ttf']
    }
    
    def __init__(self):
        """Initialize logo generator"""
        self.font_cache = {}
    
    def generate_logo(self, 
                     company_name: str, 
                     category: str = 'default',
                     style: str = 'auto',
                     size: Tuple[int, int] = (200, 80)) -> str:
        """
        Generate a deterministic, category-aware logo
        
        Args:
            company_name: Company name for logo
            category: Business category (fashion, electronics, etc.)
            style: Logo style (badge, wordmark, icon, geometric, minimal, auto)
            size: (width, height) in pixels
            
        Returns:
            Base64 encoded PNG string with data URI prefix
        """
        # Generate deterministic seed from company name
        seed = self._get_deterministic_seed(company_name)
        rng = random.Random(seed)
        
        # Select style if auto
        if style == 'auto':
            style = rng.choice(['badge', 'wordmark', 'icon', 'geometric', 'minimal'])
        
        # Get category palette
        palette = self._get_palette(category, rng)
        
        # Generate logo based on style
        if style == 'badge':
            return self._generate_badge_logo(company_name, size, palette, rng)
        elif style == 'wordmark':
            return self._generate_wordmark_logo(company_name, size, palette, rng)
        elif style == 'icon':
            return self._generate_icon_logo(company_name, size, palette, rng)
        elif style == 'geometric':
            return self._generate_geometric_logo(company_name, size, palette, rng)
        else:  # minimal
            return self._generate_minimal_logo(company_name, size, palette, rng)
    
    def _get_deterministic_seed(self, company_name: str) -> int:
        """Generate consistent seed from company name for reproducibility"""
        return int(hashlib.md5(company_name.encode('utf-8')).hexdigest()[:8], 16)
    
    def _get_palette(self, category: str, rng: random.Random) -> List[Tuple[int, int, int]]:
        """Get color palette for category"""
        category_key = category if category in self.CATEGORY_PALETTES else 'default'
        palettes = self.CATEGORY_PALETTES[category_key]
        
        # Pick a sub-palette
        palette_name = rng.choice(list(palettes.keys()))
        return palettes[palette_name]
    
    def _load_font(self, size: int, category: str = 'default') -> ImageFont.FreeTypeFont:
        """Load font with fallback support"""
        cache_key = f"{category}_{size}"
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font_prefs = self.FONT_PREFERENCES.get(category, self.FONT_PREFERENCES['default'])
        
        # Try each font in preference order
        for font_name in font_prefs:
            try:
                font = ImageFont.truetype(font_name, size)
                self.font_cache[cache_key] = font
                return font
            except (IOError, OSError):
                continue
        
        # Fallback to default
        try:
            font = ImageFont.truetype('arial.ttf', size)
            self.font_cache[cache_key] = font
            return font
        except (IOError, OSError):
            return ImageFont.load_default()
    
    def _generate_badge_logo(self, company_name: str, size: Tuple[int, int], 
                            palette: List[Tuple], rng: random.Random) -> str:
        """Generate badge-style logo with icon + text"""
        width, height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Get initials (first 2 letters of first 2 words)
        words = company_name.split()
        initials = ''.join([w[0] for w in words[:2]]).upper()[:2]
        
        # Colors
        primary, secondary, _ = palette
        
        # Badge circle/square
        badge_size = int(height * 0.75)
        margin = (height - badge_size) // 2
        shape_type = rng.choice(['circle', 'square', 'rounded'])
        
        if shape_type == 'circle':
            draw.ellipse([margin, margin, margin + badge_size, margin + badge_size], fill=primary)
        elif shape_type == 'rounded':
            # Rounded rectangle approximation
            radius = badge_size // 6
            self._draw_rounded_rectangle(draw, [margin, margin, margin + badge_size, margin + badge_size], 
                                        radius, fill=primary)
        else:
            draw.rectangle([margin, margin, margin + badge_size, margin + badge_size], fill=primary)
        
        # Initials in badge
        initial_font = self._load_font(int(badge_size * 0.45), 'default')
        bbox = draw.textbbox((0, 0), initials, font=initial_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = margin + (badge_size - text_w) // 2
        text_y = margin + (badge_size - text_h) // 2 - bbox[1]
        draw.text((text_x, text_y), initials, fill=(255, 255, 255), font=initial_font)
        
        # Company name text
        text_start = margin + badge_size + int(height * 0.2)
        available_width = width - text_start - margin
        
        # Fit text to available width
        font_size = int(height * 0.35)
        name_font = self._load_font(font_size, 'default')
        
        # Truncate if needed
        display_name = company_name
        while True:
            bbox = draw.textbbox((0, 0), display_name, font=name_font)
            if bbox[2] - bbox[0] <= available_width or font_size <= 10:
                break
            font_size -= 1
            name_font = self._load_font(font_size, 'default')
        
        bbox = draw.textbbox((0, 0), display_name, font=name_font)
        text_h = bbox[3] - bbox[1]
        text_y = (height - text_h) // 2 - bbox[1]
        draw.text((text_start, text_y), display_name, fill=primary, font=name_font)
        
        return self._image_to_base64(img)
    
    def _generate_wordmark_logo(self, company_name: str, size: Tuple[int, int],
                               palette: List[Tuple], rng: random.Random) -> str:
        """Generate text-only wordmark logo"""
        width, height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        primary, secondary, _ = palette
        
        # Large text
        font_size = int(height * 0.5)
        font = self._load_font(font_size, 'default')
        
        # Fit text
        display_name = company_name.upper()
        while True:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            if bbox[2] - bbox[0] <= width * 0.95 or font_size <= 10:
                break
            font_size -= 1
            font = self._load_font(font_size, 'default')
        
        # Center text
        bbox = draw.textbbox((0, 0), display_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (width - text_w) // 2
        text_y = (height - text_h) // 2 - bbox[1] - int(height * 0.1)
        
        draw.text((text_x, text_y), display_name, fill=primary, font=font)
        
        # Underline
        line_y = text_y + text_h + int(height * 0.05)
        line_width = int(text_w * 0.6)
        line_x = text_x + (text_w - line_width) // 2
        draw.rectangle([line_x, line_y, line_x + line_width, line_y + 3], fill=secondary)
        
        return self._image_to_base64(img)
    
    def _generate_icon_logo(self, company_name: str, size: Tuple[int, int],
                           palette: List[Tuple], rng: random.Random) -> str:
        """Generate icon above text logo"""
        width, height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        primary, secondary, _ = palette
        
        # Icon at top
        icon_size = int(height * 0.45)
        icon_x = (width - icon_size) // 2
        icon_y = int(height * 0.05)
        
        # Get initials
        words = company_name.split()
        initials = ''.join([w[0] for w in words[:2]]).upper()[:2]
        
        # Draw icon shape
        shape = rng.choice(['circle', 'diamond', 'hexagon'])
        if shape == 'circle':
            draw.ellipse([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], fill=primary)
        elif shape == 'diamond':
            self._draw_diamond(draw, [icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], primary)
        else:
            self._draw_hexagon(draw, [icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], primary)
        
        # Initials in icon
        initial_font = self._load_font(int(icon_size * 0.4), 'default')
        bbox = draw.textbbox((0, 0), initials, font=initial_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = icon_x + (icon_size - text_w) // 2
        text_y = icon_y + (icon_size - text_h) // 2 - bbox[1]
        draw.text((text_x, text_y), initials, fill=(255, 255, 255), font=initial_font)
        
        # Company name below
        text_y = icon_y + icon_size + int(height * 0.08)
        font_size = int(height * 0.25)
        font = self._load_font(font_size, 'default')
        
        display_name = company_name
        while True:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            if bbox[2] - bbox[0] <= width * 0.9 or font_size <= 8:
                break
            font_size -= 1
            font = self._load_font(font_size, 'default')
        
        bbox = draw.textbbox((0, 0), display_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = (width - text_w) // 2
        draw.text((text_x, text_y), display_name, fill=primary, font=font)
        
        return self._image_to_base64(img)
    
    def _generate_geometric_logo(self, company_name: str, size: Tuple[int, int],
                                palette: List[Tuple], rng: random.Random) -> str:
        """Generate geometric shape with text logo"""
        width, height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        primary, secondary, _ = palette
        
        # Geometric shape on left
        shape_size = int(height * 0.7)
        margin = (height - shape_size) // 2
        
        # Get initials
        words = company_name.split()
        initials = ''.join([w[0] for w in words[:2]]).upper()[:2]
        
        # Draw shape
        shape = rng.choice(['triangle', 'hexagon', 'pentagon'])
        if shape == 'triangle':
            self._draw_triangle(draw, [margin, margin, margin + shape_size, margin + shape_size], primary)
        elif shape == 'hexagon':
            self._draw_hexagon(draw, [margin, margin, margin + shape_size, margin + shape_size], primary)
        else:
            self._draw_pentagon(draw, [margin, margin, margin + shape_size, margin + shape_size], primary)
        
        # Company name
        text_start = margin + shape_size + int(height * 0.15)
        font_size = int(height * 0.4)
        font = self._load_font(font_size, 'default')
        
        display_name = company_name
        available_width = width - text_start - margin
        while True:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            if bbox[2] - bbox[0] <= available_width or font_size <= 10:
                break
            font_size -= 1
            font = self._load_font(font_size, 'default')
        
        bbox = draw.textbbox((0, 0), display_name, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = (height - text_h) // 2 - bbox[1]
        draw.text((text_start, text_y), display_name, fill=primary, font=font)
        
        return self._image_to_base64(img)
    
    def _generate_minimal_logo(self, company_name: str, size: Tuple[int, int],
                              palette: List[Tuple], rng: random.Random) -> str:
        """Generate minimal text with accent logo"""
        width, height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        primary, secondary, _ = palette
        
        # Text
        font_size = int(height * 0.45)
        font = self._load_font(font_size, 'default')
        
        display_name = company_name
        while True:
            bbox = draw.textbbox((0, 0), display_name, font=font)
            if bbox[2] - bbox[0] <= width * 0.85 or font_size <= 10:
                break
            font_size -= 1
            font = self._load_font(font_size, 'default')
        
        bbox = draw.textbbox((0, 0), display_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = int(width * 0.05)
        text_y = (height - text_h) // 2 - bbox[1]
        
        draw.text((text_x, text_y), display_name, fill=primary, font=font)
        
        # Accent (dot, line, or square)
        accent_type = rng.choice(['dot', 'line', 'square'])
        accent_x = text_x + text_w + int(width * 0.05)
        accent_size = int(height * 0.15)
        
        if accent_type == 'dot':
            draw.ellipse([accent_x, text_y + text_h // 2 - accent_size // 2,
                         accent_x + accent_size, text_y + text_h // 2 + accent_size // 2],
                        fill=secondary)
        elif accent_type == 'square':
            draw.rectangle([accent_x, text_y + text_h // 2 - accent_size // 2,
                           accent_x + accent_size, text_y + text_h // 2 + accent_size // 2],
                          fill=secondary)
        else:  # line
            line_length = int(width * 0.1)
            draw.rectangle([accent_x, text_y + text_h // 2 - 2,
                           accent_x + line_length, text_y + text_h // 2 + 2],
                          fill=secondary)
        
        return self._image_to_base64(img)
    
    # Helper drawing methods
    def _draw_rounded_rectangle(self, draw, bounds, radius, **kwargs):
        """Draw rounded rectangle"""
        x1, y1, x2, y2 = bounds
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], **kwargs)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], **kwargs)
        draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], **kwargs)
        draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], **kwargs)
        draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], **kwargs)
        draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], **kwargs)
    
    def _draw_diamond(self, draw, bounds, color):
        """Draw diamond shape"""
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        draw.polygon(points, fill=color)
    
    def _draw_hexagon(self, draw, bounds, color):
        """Draw hexagon shape"""
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = (x2 - x1) // 2
        h = (y2 - y1) // 2
        points = [
            (cx, y1),
            (x2, cy - h // 3),
            (x2, cy + h // 3),
            (cx, y2),
            (x1, cy + h // 3),
            (x1, cy - h // 3)
        ]
        draw.polygon(points, fill=color)
    
    def _draw_triangle(self, draw, bounds, color):
        """Draw triangle shape"""
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) // 2
        points = [(cx, y1), (x2, y2), (x1, y2)]
        draw.polygon(points, fill=color)
    
    def _draw_pentagon(self, draw, bounds, color):
        """Draw pentagon shape"""
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        import math
        radius = min((x2 - x1), (y2 - y1)) // 2
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


if __name__ == "__main__":
    # Test generation
    gen = LogoGenerator()
    
    print("Testing Logo Generator...")
    print("=" * 60)
    
    test_companies = [
        ("Acme Corp", "electronics"),
        ("Fashion Boutique", "fashion"),
        ("Beauty Haven", "beauty"),
        ("Tech Solutions", "electronics"),
        ("Garden Center", "home_garden")
    ]
    
    for company, category in test_companies:
        logo = gen.generate_logo(company, category)
        print(f"✓ Generated logo for {company} ({category})")
        print(f"  Length: {len(logo)} bytes")
    
    print("\n" + "=" * 60)
    print("Testing determinism...")
    logo1 = gen.generate_logo("Test Co", "fashion")
    logo2 = gen.generate_logo("Test Co", "fashion")
    if logo1 == logo2:
        print("✓ Deterministic: Same company produces identical logo")
    else:
        print("✗ Non-deterministic: Logos differ")
    
    print("\nLogo Generator Ready!")
