"""
Shopify Product CSV Generator
Generates Shopify-compatible product import CSV files with realistic data.

Supports:
- Multi-variant products (Size, Color, Material combinations)
- All required Shopify CSV columns (30+)
- Multiple product categories (Fashion, Electronics, Home, Beauty, etc.)
- Realistic product names, descriptions, prices, and SKUs
- Image URL placeholders or real placeholder images
- SEO fields (title, description)
- Inventory tracking
- Weight and shipping info
"""

import csv
import random
import string
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from faker import Faker


@dataclass
class ProductVariant:
    """Represents a single product variant."""
    option1_value: str = ""
    option2_value: str = ""
    option3_value: str = ""
    sku: str = ""
    barcode: str = ""
    price: float = 0.0
    compare_at_price: Optional[float] = None
    cost_per_item: Optional[float] = None
    inventory_qty: int = 0
    weight_grams: int = 0
    variant_image_url: Optional[str] = None


@dataclass 
class ShopifyProduct:
    """Represents a complete Shopify product with variants."""
    handle: str = ""
    title: str = ""
    description: str = ""
    vendor: str = ""
    product_category: str = ""
    product_type: str = ""
    tags: List[str] = field(default_factory=list)
    published: bool = True
    status: str = "active"
    
    # Options (up to 3)
    option1_name: str = "Title"
    option2_name: Optional[str] = None
    option3_name: Optional[str] = None
    
    # Variants
    variants: List[ProductVariant] = field(default_factory=list)
    
    # Images
    image_urls: List[str] = field(default_factory=list)
    image_alt_texts: List[str] = field(default_factory=list)
    
    # Shipping
    requires_shipping: bool = True
    weight_unit: str = "g"
    
    # Tax
    taxable: bool = True
    
    # SEO
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    
    # Inventory
    inventory_tracker: str = "shopify"
    inventory_policy: str = "deny"  # deny or continue
    fulfillment_service: str = "manual"
    
    # Gift card
    gift_card: bool = False
    
    # Collection (optional)
    collection: Optional[str] = None


class ShopifyProductCSVGenerator:
    """Generates Shopify-compatible product CSV files."""
    
    # Shopify CSV column headers (official format)
    CSV_HEADERS = [
        "Handle",
        "Title", 
        "Body (HTML)",
        "Vendor",
        "Product Category",
        "Type",
        "Tags",
        "Published",
        "Option1 Name",
        "Option1 Value",
        "Option2 Name", 
        "Option2 Value",
        "Option3 Name",
        "Option3 Value",
        "Variant SKU",
        "Variant Grams",
        "Variant Inventory Tracker",
        "Variant Inventory Qty",
        "Variant Inventory Policy",
        "Variant Fulfillment Service",
        "Variant Price",
        "Variant Compare At Price",
        "Variant Requires Shipping",
        "Variant Taxable",
        "Variant Barcode",
        "Image Src",
        "Image Position",
        "Image Alt Text",
        "Gift Card",
        "SEO Title",
        "SEO Description",
        "Variant Image",
        "Variant Weight Unit",
        "Variant Tax Code",
        "Cost per item",
        "Included / Primary",
        "Included / International",
        "Status",
        "Collection",
    ]
    
    # Product categories with Shopify taxonomy
    PRODUCT_CATEGORIES = {
        "apparel": {
            "category": "Apparel & Accessories > Clothing",
            "types": ["T-Shirts", "Jeans", "Dresses", "Hoodies", "Jackets", "Sweaters", "Shorts", "Skirts"],
            "options": {
                "Size": ["XS", "S", "M", "L", "XL", "XXL"],
                "Color": ["Black", "White", "Navy", "Gray", "Red", "Blue", "Green", "Pink", "Beige"],
            },
            "weight_range": (150, 800),
            "price_range": (19.99, 149.99),
        },
        "shoes": {
            "category": "Apparel & Accessories > Shoes",
            "types": ["Sneakers", "Boots", "Sandals", "Loafers", "Heels", "Flats", "Athletic Shoes"],
            "options": {
                "Size": ["6", "7", "8", "9", "10", "11", "12", "13"],
                "Color": ["Black", "White", "Brown", "Tan", "Navy", "Gray"],
            },
            "weight_range": (300, 1200),
            "price_range": (49.99, 249.99),
        },
        "electronics": {
            "category": "Electronics",
            "types": ["Headphones", "Speakers", "Phone Cases", "Chargers", "Cables", "Power Banks"],
            "options": {
                "Color": ["Black", "White", "Silver", "Space Gray", "Rose Gold"],
            },
            "weight_range": (50, 500),
            "price_range": (14.99, 299.99),
        },
        "home_decor": {
            "category": "Home & Garden > Decor",
            "types": ["Candles", "Vases", "Picture Frames", "Throw Pillows", "Wall Art", "Rugs"],
            "options": {
                "Size": ["Small", "Medium", "Large"],
                "Color": ["White", "Black", "Natural", "Gray", "Blush", "Sage"],
            },
            "weight_range": (200, 2000),
            "price_range": (24.99, 199.99),
        },
        "beauty": {
            "category": "Health & Beauty > Personal Care",
            "types": ["Moisturizer", "Serum", "Cleanser", "Lipstick", "Foundation", "Mascara"],
            "options": {
                "Size": ["Travel Size", "Regular", "Value Size"],
                "Shade": ["Fair", "Light", "Medium", "Tan", "Deep"],
            },
            "weight_range": (30, 250),
            "price_range": (12.99, 89.99),
        },
        "jewelry": {
            "category": "Apparel & Accessories > Jewelry",
            "types": ["Necklaces", "Earrings", "Bracelets", "Rings", "Anklets"],
            "options": {
                "Material": ["Gold", "Silver", "Rose Gold", "Platinum"],
                "Size": ["One Size", "Adjustable", "Small", "Medium", "Large"],
            },
            "weight_range": (5, 100),
            "price_range": (29.99, 499.99),
        },
        "bags": {
            "category": "Apparel & Accessories > Handbags, Wallets & Cases",
            "types": ["Tote Bags", "Backpacks", "Crossbody Bags", "Clutches", "Wallets"],
            "options": {
                "Color": ["Black", "Brown", "Tan", "Navy", "Burgundy", "Cream"],
                "Size": ["Small", "Medium", "Large"],
            },
            "weight_range": (200, 1000),
            "price_range": (39.99, 299.99),
        },
        "fitness": {
            "category": "Sporting Goods > Exercise & Fitness",
            "types": ["Yoga Mats", "Resistance Bands", "Dumbbells", "Water Bottles", "Gym Bags"],
            "options": {
                "Color": ["Black", "Blue", "Pink", "Purple", "Green"],
                "Size": ["One Size", "Small", "Medium", "Large"],
            },
            "weight_range": (100, 5000),
            "price_range": (14.99, 149.99),
        },
        "pet_supplies": {
            "category": "Animals & Pet Supplies",
            "types": ["Dog Toys", "Cat Toys", "Pet Beds", "Collars", "Leashes", "Food Bowls"],
            "options": {
                "Size": ["Small", "Medium", "Large", "XL"],
                "Color": ["Blue", "Red", "Pink", "Green", "Black"],
            },
            "weight_range": (50, 2000),
            "price_range": (9.99, 79.99),
        },
        "kitchen": {
            "category": "Home & Garden > Kitchen & Dining",
            "types": ["Mugs", "Cutting Boards", "Utensil Sets", "Storage Containers", "Bakeware"],
            "options": {
                "Color": ["White", "Black", "Gray", "Natural Wood", "Stainless Steel"],
                "Size": ["Small", "Medium", "Large", "Set"],
            },
            "weight_range": (100, 1500),
            "price_range": (14.99, 99.99),
        },
        # ============ NEW CATEGORIES (Phase 2 Expansion) ============
        "candy": {
            "category": "Food, Beverages & Tobacco > Food Items > Confectionery",
            "types": ["Chocolate Bars", "Gummy Bears", "Hard Candy", "Lollipops", "Caramels", 
                      "Licorice", "Mints", "Candy Canes", "Jelly Beans", "Taffy"],
            "options": {
                "Flavor": ["Original", "Strawberry", "Chocolate", "Mint", "Mixed Fruit", "Sour", "Caramel"],
                "Size": ["Single", "Snack Pack", "Share Size", "Family Pack", "Bulk Bag"],
            },
            "weight_range": (30, 500),
            "price_range": (2.99, 24.99),
        },
        "books_media": {
            "category": "Media > Books",
            "types": ["Hardcover Books", "Paperback Books", "Journals", "Planners", "Notebooks", 
                      "Coloring Books", "Cookbooks", "Art Books", "Children's Books", "Comics"],
            "options": {
                "Format": ["Hardcover", "Paperback", "Spiral Bound"],
                "Edition": ["Standard", "Deluxe", "Collector's", "Limited Edition"],
            },
            "weight_range": (150, 1200),
            "price_range": (9.99, 49.99),
        },
        "toys_games": {
            "category": "Toys & Games",
            "types": ["Board Games", "Puzzles", "Action Figures", "Dolls", "Building Blocks", 
                      "Remote Control Cars", "Stuffed Animals", "Card Games", "Educational Toys", "Outdoor Toys"],
            "options": {
                "Age Group": ["0-2 Years", "3-5 Years", "6-8 Years", "9-12 Years", "Teen", "Adult"],
                "Theme": ["Classic", "Adventure", "Fantasy", "Educational", "Sports"],
            },
            "weight_range": (100, 2000),
            "price_range": (12.99, 79.99),
        },
        "sports_equipment": {
            "category": "Sporting Goods",
            "types": ["Basketballs", "Soccer Balls", "Tennis Rackets", "Golf Clubs", "Baseball Bats", 
                      "Hockey Sticks", "Footballs", "Volleyball Sets", "Badminton Sets", "Skateboards"],
            "options": {
                "Size": ["Youth", "Junior", "Standard", "Professional"],
                "Color": ["Black", "White", "Blue", "Red", "Orange", "Green"],
            },
            "weight_range": (200, 3000),
            "price_range": (19.99, 199.99),
        },
        "automotive": {
            "category": "Vehicles & Parts > Vehicle Parts & Accessories",
            "types": ["Car Phone Mounts", "Seat Covers", "Floor Mats", "Air Fresheners", "Dash Cams", 
                      "Jump Starters", "Tire Gauges", "Car Chargers", "Sunshades", "Steering Wheel Covers"],
            "options": {
                "Fit": ["Universal", "Compact", "Sedan", "SUV", "Truck"],
                "Color": ["Black", "Gray", "Beige", "Brown", "Red"],
            },
            "weight_range": (50, 2500),
            "price_range": (9.99, 149.99),
        },
        "office_supplies": {
            "category": "Office Supplies",
            "types": ["Pens", "Notebooks", "Staplers", "Desk Organizers", "File Folders", 
                      "Sticky Notes", "Highlighters", "Paper Clips", "Scissors", "Tape Dispensers"],
            "options": {
                "Color": ["Black", "Blue", "Red", "Assorted", "Clear"],
                "Pack Size": ["Single", "3-Pack", "6-Pack", "12-Pack", "Bulk"],
            },
            "weight_range": (20, 500),
            "price_range": (4.99, 39.99),
        },
        "garden_outdoor": {
            "category": "Home & Garden > Lawn & Garden",
            "types": ["Plant Pots", "Garden Tools", "Seeds", "Watering Cans", "Outdoor Lights", 
                      "Bird Feeders", "Planters", "Garden Gloves", "Hose Nozzles", "Lawn Ornaments"],
            "options": {
                "Size": ["Small", "Medium", "Large", "XL"],
                "Material": ["Ceramic", "Plastic", "Metal", "Wood", "Terra Cotta"],
            },
            "weight_range": (100, 3000),
            "price_range": (7.99, 89.99),
        },
        "baby_products": {
            "category": "Baby & Toddler",
            "types": ["Onesies", "Bibs", "Bottles", "Pacifiers", "Diapers", 
                      "Blankets", "Teethers", "Baby Wipes", "Diaper Bags", "Baby Monitors"],
            "options": {
                "Size": ["Newborn", "0-3 Months", "3-6 Months", "6-12 Months", "12-18 Months", "18-24 Months"],
                "Color": ["White", "Pink", "Blue", "Yellow", "Green", "Neutral"],
            },
            "weight_range": (30, 800),
            "price_range": (6.99, 79.99),
        },
    }
    
    # Brand name components for generating realistic vendor names
    BRAND_PREFIXES = [
        "Urban", "Pure", "Eco", "Luxe", "Modern", "Classic", "Premier", "Elite",
        "Nordic", "Pacific", "Alpine", "Golden", "Silver", "Crystal", "Royal",
        "Artisan", "Heritage", "Vintage", "Nova", "Aurora", "Stellar", "Apex",
    ]
    
    BRAND_SUFFIXES = [
        "Co", "Studio", "House", "Lab", "Collective", "Works", "Atelier",
        "Design", "Living", "Home", "Goods", "Supply", "Essentials", "Basics",
        "Made", "Craft", "Style", "Shop", "Store", "Brand", "Line",
    ]
    
    # Product name templates by category
    PRODUCT_TEMPLATES = {
        "apparel": [
            "{adj} {material} {type}",
            "{brand} {adj} {type}",
            "{material} {type} - {style}",
            "Premium {material} {type}",
            "{style} {type} Collection",
        ],
        "default": [
            "{adj} {type}",
            "{brand} {type}",
            "Premium {type}",
            "{type} - {style} Edition",
            "Classic {type}",
        ],
    }
    
    ADJECTIVES = [
        "Premium", "Classic", "Modern", "Vintage", "Elegant", "Minimalist",
        "Luxurious", "Cozy", "Sleek", "Essential", "Signature", "Everyday",
        "Ultimate", "Professional", "Organic", "Natural", "Handcrafted",
    ]
    
    MATERIALS = {
        "apparel": ["Cotton", "Linen", "Silk", "Wool", "Cashmere", "Denim", "Jersey", "Fleece"],
        "shoes": ["Leather", "Suede", "Canvas", "Mesh", "Synthetic"],
        "bags": ["Leather", "Canvas", "Vegan Leather", "Nylon", "Cotton"],
        "jewelry": ["Sterling Silver", "14K Gold", "Rose Gold", "Platinum", "Stainless Steel"],
        "default": ["Premium", "Quality", "Durable", "Sustainable"],
    }
    
    STYLES = [
        "Essential", "Signature", "Heritage", "Modern", "Classic", "Everyday",
        "Limited Edition", "Seasonal", "Core", "Pro", "Original",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator."""
        self.fake = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
    
    def generate_handle(self, title: str) -> str:
        """Generate URL-safe handle from title."""
        handle = title.lower()
        # Replace special characters
        for char in ["'", '"', "&", "/", "\\", "(", ")", "[", "]", "{", "}", ",", ".", "!"]:
            handle = handle.replace(char, "")
        # Replace spaces and multiple dashes
        handle = handle.replace(" ", "-")
        while "--" in handle:
            handle = handle.replace("--", "-")
        handle = handle.strip("-")
        return handle[:50]  # Limit length
    
    def generate_sku(self, prefix: str = "") -> str:
        """Generate a unique SKU."""
        chars = ''.join(random.choices(string.ascii_uppercase, k=3))
        nums = ''.join(random.choices(string.digits, k=5))
        return f"{prefix}{chars}-{nums}" if prefix else f"{chars}-{nums}"
    
    def generate_barcode(self) -> str:
        """Generate a realistic UPC/EAN barcode."""
        # Generate 12-digit UPC-A
        return ''.join(random.choices(string.digits, k=12))
    
    def generate_vendor(self) -> str:
        """Generate a realistic brand/vendor name."""
        if random.random() < 0.3:
            # Use Faker company name
            return self.fake.company().split()[0] + " " + random.choice(self.BRAND_SUFFIXES)
        else:
            # Generate from components
            prefix = random.choice(self.BRAND_PREFIXES)
            suffix = random.choice(self.BRAND_SUFFIXES)
            return f"{prefix} {suffix}"
    
    def generate_description(self, product_type: str, category: str, 
                            options: Dict[str, List[str]]) -> str:
        """Generate HTML product description."""
        features = [
            f"High-quality {product_type.lower()} for everyday use",
            "Designed with attention to detail",
            "Made from premium materials",
            "Perfect for any occasion",
        ]
        
        if category in ["apparel", "shoes"]:
            features.extend([
                "Comfortable fit guaranteed",
                "Machine washable for easy care",
                "Available in multiple sizes and colors",
            ])
        elif category == "electronics":
            features.extend([
                "Long-lasting battery life",
                "Compatible with all major devices",
                "1-year manufacturer warranty included",
            ])
        elif category in ["home_decor", "kitchen"]:
            features.extend([
                "Adds style to any space",
                "Easy to clean and maintain",
                "Durable construction for lasting use",
            ])
        
        selected_features = random.sample(features, min(4, len(features)))
        
        html = f"<p>Discover our {product_type.lower()} - the perfect addition to your collection.</p>\n"
        html += "<ul>\n"
        for feature in selected_features:
            html += f"  <li>{feature}</li>\n"
        html += "</ul>\n"
        html += f"<p>Available options: {', '.join(options.keys())}</p>"
        
        return html
    
    def generate_tags(self, category: str, product_type: str, vendor: str) -> List[str]:
        """Generate relevant product tags."""
        tags = [
            category.replace("_", "-"),
            product_type.lower().replace(" ", "-"),
            vendor.lower().replace(" ", "-"),
        ]
        
        # Add seasonal/trending tags
        seasonal = ["new-arrival", "trending", "bestseller", "staff-pick"]
        if random.random() < 0.3:
            tags.append(random.choice(seasonal))
        
        # Add category-specific tags
        category_tags = {
            "apparel": ["fashion", "clothing", "wardrobe-essential"],
            "shoes": ["footwear", "shoes", "comfort"],
            "electronics": ["tech", "gadgets", "accessories"],
            "beauty": ["skincare", "beauty", "self-care"],
            "home_decor": ["home", "decor", "interior"],
            "jewelry": ["accessories", "jewelry", "gifts"],
        }
        
        if category in category_tags:
            tags.extend(random.sample(category_tags[category], min(2, len(category_tags[category]))))
        
        return list(set(tags))[:10]  # Max 10 unique tags
    
    def generate_image_url(self, product_type: str, index: int = 1) -> str:
        """Generate a placeholder image URL."""
        # Use placeholder image service
        width, height = 800, 800
        # Using picsum.photos for realistic placeholder images
        seed = hash(f"{product_type}-{index}") % 1000
        return f"https://picsum.photos/seed/{seed}/{width}/{height}"
    
    def generate_product(self, category: str = None, 
                        num_variants: int = None,
                        vendor: str = None) -> ShopifyProduct:
        """Generate a complete Shopify product."""
        
        # Select category
        if category is None:
            category = random.choice(list(self.PRODUCT_CATEGORIES.keys()))
        
        cat_config = self.PRODUCT_CATEGORIES.get(category, self.PRODUCT_CATEGORIES["apparel"])
        
        # Select product type
        product_type = random.choice(cat_config["types"])
        
        # Generate vendor
        if vendor is None:
            vendor = self.generate_vendor()
        
        # Generate product name
        adj = random.choice(self.ADJECTIVES)
        material = random.choice(self.MATERIALS.get(category, self.MATERIALS["default"]))
        style = random.choice(self.STYLES)
        
        title_templates = self.PRODUCT_TEMPLATES.get(category, self.PRODUCT_TEMPLATES["default"])
        title_template = random.choice(title_templates)
        title = title_template.format(
            adj=adj, material=material, type=product_type, 
            style=style, brand=vendor.split()[0]
        )
        
        # Generate handle
        handle = self.generate_handle(title)
        
        # Determine options and variants
        available_options = cat_config.get("options", {})
        option_names = list(available_options.keys())[:3]  # Max 3 options
        
        # Determine number of variants
        if num_variants is None:
            if len(option_names) >= 2:
                num_variants = random.randint(4, 12)
            elif len(option_names) == 1:
                num_variants = random.randint(3, 6)
            else:
                num_variants = 1
        
        # Generate variants
        variants = []
        used_combinations = set()
        
        option1_name = option_names[0] if option_names else "Title"
        option2_name = option_names[1] if len(option_names) > 1 else None
        option3_name = option_names[2] if len(option_names) > 2 else None
        
        option1_values = available_options.get(option1_name, ["Default"])
        option2_values = available_options.get(option2_name, [""]) if option2_name else [""]
        option3_values = available_options.get(option3_name, [""]) if option3_name else [""]
        
        # Generate base price
        price_min, price_max = cat_config.get("price_range", (19.99, 99.99))
        base_price = round(random.uniform(price_min, price_max), 2)
        
        # Ensure price ends in .99 or .00
        base_price = round(base_price) - 0.01 if random.random() < 0.7 else round(base_price)
        
        # Generate weight range
        weight_min, weight_max = cat_config.get("weight_range", (100, 500))
        base_weight = random.randint(weight_min, weight_max)
        
        # Generate variants
        attempts = 0
        while len(variants) < num_variants and attempts < 100:
            attempts += 1
            
            opt1 = random.choice(option1_values)
            opt2 = random.choice(option2_values) if option2_name else ""
            opt3 = random.choice(option3_values) if option3_name else ""
            
            combo = (opt1, opt2, opt3)
            if combo in used_combinations:
                continue
            used_combinations.add(combo)
            
            # Slight price variation per variant
            variant_price = base_price + random.choice([0, 5, 10, -5]) if random.random() < 0.3 else base_price
            variant_price = max(9.99, variant_price)  # Minimum price
            
            # Compare at price (for sales)
            compare_at = None
            if random.random() < 0.25:  # 25% chance of sale
                compare_at = round(variant_price * random.uniform(1.2, 1.5), 2)
            
            # Cost per item (wholesale cost)
            cost = round(variant_price * random.uniform(0.3, 0.5), 2)
            
            # Weight variation
            variant_weight = base_weight + random.randint(-50, 50)
            variant_weight = max(10, variant_weight)
            
            # Inventory
            inventory = random.choices(
                [0, random.randint(1, 10), random.randint(10, 50), random.randint(50, 200)],
                weights=[0.05, 0.15, 0.40, 0.40]
            )[0]
            
            variant = ProductVariant(
                option1_value=opt1,
                option2_value=opt2,
                option3_value=opt3,
                sku=self.generate_sku(handle[:3].upper()),
                barcode=self.generate_barcode(),
                price=variant_price,
                compare_at_price=compare_at,
                cost_per_item=cost,
                inventory_qty=inventory,
                weight_grams=variant_weight,
            )
            variants.append(variant)
        
        # If no variants generated, add default
        if not variants:
            variants.append(ProductVariant(
                option1_value="Default Title",
                sku=self.generate_sku(handle[:3].upper()),
                barcode=self.generate_barcode(),
                price=base_price,
                inventory_qty=random.randint(10, 100),
                weight_grams=base_weight,
            ))
            option1_name = "Title"
            option2_name = None
            option3_name = None
        
        # Generate description
        description = self.generate_description(product_type, category, available_options)
        
        # Generate tags
        tags = self.generate_tags(category, product_type, vendor)
        
        # Generate images (2-4 per product)
        num_images = random.randint(2, 4)
        image_urls = [self.generate_image_url(product_type, i+1) for i in range(num_images)]
        image_alt_texts = [f"{title} - Image {i+1}" for i in range(num_images)]
        
        # SEO
        seo_title = f"{title} | {vendor}" if len(title) < 50 else title[:60]
        seo_desc = f"Shop {title} from {vendor}. {random.choice(self.ADJECTIVES)} {product_type.lower()} at great prices. Free shipping available."
        
        # Collection
        collection = product_type if random.random() < 0.5 else None
        
        return ShopifyProduct(
            handle=handle,
            title=title,
            description=description,
            vendor=vendor,
            product_category=cat_config["category"],
            product_type=product_type,
            tags=tags,
            published=True,
            status="active",
            option1_name=option1_name,
            option2_name=option2_name,
            option3_name=option3_name,
            variants=variants,
            image_urls=image_urls,
            image_alt_texts=image_alt_texts,
            requires_shipping=True,
            weight_unit="g",
            taxable=True,
            seo_title=seo_title,
            seo_description=seo_desc,
            inventory_tracker="shopify",
            inventory_policy="deny",
            fulfillment_service="manual",
            gift_card=False,
            collection=collection,
        )
    
    def product_to_csv_rows(self, product: ShopifyProduct) -> List[Dict[str, Any]]:
        """Convert a ShopifyProduct to CSV rows (one per variant + images)."""
        rows = []
        
        for i, variant in enumerate(product.variants):
            row = {
                "Handle": product.handle,
                "Title": product.title if i == 0 else "",
                "Body (HTML)": product.description if i == 0 else "",
                "Vendor": product.vendor if i == 0 else "",
                "Product Category": product.product_category if i == 0 else "",
                "Type": product.product_type if i == 0 else "",
                "Tags": ", ".join(product.tags) if i == 0 else "",
                "Published": "true" if product.published else "false",
                "Option1 Name": product.option1_name if i == 0 else "",
                "Option1 Value": variant.option1_value,
                "Option2 Name": product.option2_name or "" if i == 0 else "",
                "Option2 Value": variant.option2_value,
                "Option3 Name": product.option3_name or "" if i == 0 else "",
                "Option3 Value": variant.option3_value,
                "Variant SKU": variant.sku,
                "Variant Grams": variant.weight_grams,
                "Variant Inventory Tracker": product.inventory_tracker,
                "Variant Inventory Qty": variant.inventory_qty,
                "Variant Inventory Policy": product.inventory_policy,
                "Variant Fulfillment Service": product.fulfillment_service,
                "Variant Price": variant.price,
                "Variant Compare At Price": variant.compare_at_price or "",
                "Variant Requires Shipping": "true" if product.requires_shipping else "false",
                "Variant Taxable": "true" if product.taxable else "false",
                "Variant Barcode": variant.barcode,
                "Image Src": product.image_urls[0] if i == 0 and product.image_urls else "",
                "Image Position": 1 if i == 0 and product.image_urls else "",
                "Image Alt Text": product.image_alt_texts[0] if i == 0 and product.image_alt_texts else "",
                "Gift Card": "true" if product.gift_card else "false",
                "SEO Title": product.seo_title if i == 0 else "",
                "SEO Description": product.seo_description if i == 0 else "",
                "Variant Image": variant.variant_image_url or "",
                "Variant Weight Unit": product.weight_unit,
                "Variant Tax Code": "",
                "Cost per item": variant.cost_per_item or "",
                "Included / Primary": "true",
                "Included / International": "true",
                "Status": product.status if i == 0 else "",
                "Collection": product.collection or "" if i == 0 else "",
            }
            rows.append(row)
        
        # Add additional image rows (images 2+)
        for img_idx in range(1, len(product.image_urls)):
            row = {col: "" for col in self.CSV_HEADERS}
            row["Handle"] = product.handle
            row["Image Src"] = product.image_urls[img_idx]
            row["Image Position"] = img_idx + 1
            row["Image Alt Text"] = product.image_alt_texts[img_idx] if img_idx < len(product.image_alt_texts) else ""
            rows.append(row)
        
        return rows
    
    def generate_csv(self, 
                    num_products: int = 10,
                    categories: List[str] = None,
                    output_path: str = "shopify_products.csv",
                    vendor: str = None) -> str:
        """Generate a complete Shopify product CSV file.
        
        Args:
            num_products: Number of products to generate
            categories: List of categories to use (None = random mix)
            output_path: Path for output CSV file
            vendor: Specific vendor name (None = generate random)
            
        Returns:
            Path to generated CSV file
        """
        products = []
        
        for i in range(num_products):
            category = random.choice(categories) if categories else None
            product = self.generate_product(category=category, vendor=vendor)
            products.append(product)
        
        # Convert to CSV rows
        all_rows = []
        for product in products:
            all_rows.extend(self.product_to_csv_rows(product))
        
        # Write CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            writer.writerows(all_rows)
        
        return str(output_path)
    
    def generate_sample_products(self, num_products: int = 5) -> List[ShopifyProduct]:
        """Generate a list of sample products (in-memory, no file)."""
        products = []
        for _ in range(num_products):
            category = random.choice(list(self.PRODUCT_CATEGORIES.keys()))
            products.append(self.generate_product(category=category))
        return products


def main():
    """Test the generator."""
    generator = ShopifyProductCSVGenerator(seed=42)
    
    # Generate sample CSV
    print("Generating Shopify Product CSV...")
    output_path = generator.generate_csv(
        num_products=25,
        output_path="outputs/shopify_products/sample_products.csv"
    )
    print(f"✓ Generated: {output_path}")
    
    # Show sample product info
    print("\nSample products generated:")
    products = generator.generate_sample_products(5)
    for p in products:
        print(f"  - {p.title} ({p.vendor}) - {len(p.variants)} variants, ${p.variants[0].price:.2f}")


if __name__ == "__main__":
    main()
