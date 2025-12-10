"""
Dynamic Brand Name Generator
Generates infinite realistic store/brand names based on product category patterns
"""
import random
from typing import List, Tuple


class BrandNameGenerator:
    """Generates realistic brand names using linguistic patterns and category-specific vocabulary"""
    
    # TIER 2: Descriptive prefixes for modifier-first patterns (60+ prefixes)
    # Applied to 20% of compound/single-word patterns to create variants like "Premium Cotton Studio"
    DESCRIPTIVE_PREFIXES = [
        # Quality (10)
        'Premium', 'Luxury', 'Elite', 'Prime', 'Superior', 'Select', 'Fine', 'Refined', 'Polished', 'Exquisite',
        # Style (15)
        'Modern', 'Vintage', 'Contemporary', 'Classic', 'Timeless', 'Minimalist', 'Bold', 'Elegant', 'Chic',
        'Sleek', 'Urban', 'Rustic', 'Industrial', 'Artisan', 'Boutique',
        # Emotion/Energy (10)
        'Happy', 'Joyful', 'Serene', 'Vibrant', 'Calm', 'Wild', 'Free', 'Bright', 'Radiant', 'Lively',
        # Origin/Authenticity (10)
        'True', 'Original', 'Authentic', 'Natural', 'Organic', 'Raw', 'Pure', 'Real', 'Genuine', 'Native',
        # Scale/Intensity (8)
        'Micro', 'Mini', 'Grand', 'Mega', 'Ultra', 'Super', 'Maxi', 'Epic',
        # Innovation (7)
        'Next', 'Future', 'Forward', 'New', 'Fresh', 'Novel', 'Pioneer'
    ]
    
    # Prefixes by category theme (existing system, kept for compatibility)
    PREFIXES = {
        'modern': ['Urban', 'Modern', 'Pure', 'Fresh', 'True', 'Vital', 'Peak', 'Prime', 'Core', 'Nova'],
        'luxury': ['Luxe', 'Elite', 'Royal', 'Premier', 'Grand', 'Prestige', 'Noble', 'Regal', 'Gilt', 'Opulent'],
        'natural': ['Pure', 'Natural', 'Organic', 'Wild', 'Raw', 'Earth', 'Green', 'Eco', 'Vital', 'Native'],
        'tech': ['Smart', 'Digital', 'Tech', 'Cyber', 'Data', 'Cloud', 'Sync', 'Net', 'Bit', 'Byte'],
        'artisan': ['Craft', 'Artisan', 'Hand', 'Made', 'Forge', 'Studio', 'Workshop', 'Atelier', 'Guild'],
        'classic': ['Classic', 'Heritage', 'Legacy', 'Vintage', 'Timeless', 'Traditional', 'Original', 'Authentic'],
        'energetic': ['Active', 'Dynamic', 'Vital', 'Power', 'Energy', 'Force', 'Vigor', 'Pulse', 'Rush'],
        'minimal': ['Simple', 'Clean', 'Bare', 'Minimal', 'Essential', 'Pure', 'Naked', 'Plain', 'Basic']
    }
    
    # Category-specific root words
    CATEGORY_ROOTS = {
        'fashion': {
            'nouns': ['Thread', 'Stitch', 'Fabric', 'Closet', 'Wardrobe', 'Style', 'Wear', 'Apparel', 'Cloth', 'Garment', 
                     'Attire', 'Outfit', 'Fashion', 'Vogue', 'Trend', 'Mode', 'Boutique', 'Couture', 'Tailor', 'Sartorial',
                     'Drape', 'Hem', 'Seam', 'Pattern', 'Textile', 'Weave', 'Knit', 'Loom', 'Silk', 'Cotton',
                     'Denim', 'Leather', 'Suede', 'Wool', 'Linen', 'Cashmere', 'Velvet', 'Satin', 'Tweed', 'Flannel',
                     'Collar', 'Cuff', 'Button', 'Zipper', 'Pocket', 'Sleeve', 'Lapel', 'Pleat', 'Ruffle', 'Trim',
                     'Mannequin', 'Runway', 'Catwalk', 'Atelier', 'Maison', 'House', 'Label', 'Brand', 'Collection', 'Line'],
            'adjectives': [
                # TIER 2: Expanded from 30 to 80+ adjectives for fashion (2.7x increase)
                # Original core (30)
                'Chic', 'Stylish', 'Trendy', 'Classic', 'Modern', 'Elegant', 'Bold', 'Sleek', 'Sharp', 'Refined',
                'Dapper', 'Suave', 'Polished', 'Tailored', 'Fitted', 'Bespoke', 'Custom', 'Haute', 'Pret', 'Ready',
                'Urban', 'Street', 'Casual', 'Formal', 'Smart', 'Business', 'Evening', 'Day', 'Night', 'Season',
                # Texture adjectives (20)
                'Smooth', 'Soft', 'Crisp', 'Plush', 'Velvet', 'Suede', 'Knit', 'Woven', 'Draped', 'Structured',
                'Fluid', 'Flowing', 'Tailored', 'Fitted', 'Loose', 'Tight', 'Stretch', 'Rigid', 'Supple', 'Luxe',
                # Color/Visual adjectives (15)
                'Azure', 'Crimson', 'Ivory', 'Jade', 'Amber', 'Coral', 'Pearl', 'Obsidian', 'Ruby', 'Sapphire',
                'Noir', 'Blanc', 'Indigo', 'Scarlet', 'Emerald',
                # Quality adjectives (15)
                'Fine', 'Refined', 'Polished', 'Crafted', 'Curated', 'Select', 'Premium', 'Elite', 'Luxury', 'Artisan',
                'Handmade', 'Couture', 'Designer', 'Signature', 'Exclusive'
            ],
            'themes': ['modern', 'luxury', 'minimal']
        },
        'electronics': {
            'nouns': ['Tech', 'Gadget', 'Device', 'Circuit', 'Volt', 'Wire', 'Code', 'Byte', 'Pixel', 'Screen',
                     'Digital', 'Cyber', 'Connect', 'Signal', 'Wave', 'Charge', 'Power', 'Grid', 'Socket', 'Port',
                     'Chip', 'Core', 'Processor', 'Memory', 'Drive', 'Disk', 'Flash', 'RAM', 'CPU', 'GPU',
                     'Network', 'Router', 'Switch', 'Hub', 'Node', 'Server', 'Cloud', 'Data', 'Storage', 'Backup',
                     'Display', 'Monitor', 'Panel', 'LED', 'OLED', 'LCD', 'Touch', 'Sensor', 'Camera', 'Lens',
                     'Audio', 'Speaker', 'Sound', 'Voice', 'Echo', 'Bluetooth', 'WiFi', 'USB', 'HDMI', 'Cord'],
            'adjectives': [
                # TIER 2: Expanded from 35 to 70+ adjectives for electronics (2x increase)
                # Original core (35)
                'Smart', 'Connected', 'Wireless', 'Digital', 'Advanced', 'Cutting-Edge', 'Innovative',
                'Intelligent', 'Automated', 'AI', 'Neural', 'Quantum', 'Next-Gen', 'Future', 'Pro', 'Max',
                'Ultra', 'Mega', 'Hyper', 'Super', 'Turbo', 'Rapid', 'Fast', 'Quick', 'Instant', 'Real-Time',
                'Secure', 'Protected', 'Encrypted', 'Safe', 'Reliable', 'Stable', 'Robust', 'Solid', 'Prime', 'Elite',
                # Modern tech adjectives (20)
                'Cloud', 'Edge', 'Nano', 'Micro', 'Sonic', 'Photon', 'Electron', 'Atom', 'Spark', 'Pulse',
                'Wave', 'Stream', 'Flow', 'Sync', 'Link', 'Connect', 'Network', 'Mesh', 'Grid', 'Hub',
                # Performance adjectives (15)
                'Efficient', 'Powerful', 'Mighty', 'Precision', 'Accurate', 'Sharp', 'Clear', 'Vivid', 'Bright',
                'Brilliant', 'Peak', 'Optimal', 'Maximum', 'Enhanced', 'Boosted'
            ],
            'themes': ['tech', 'modern']
        },
        'beauty': {
            'nouns': ['Glow', 'Radiance', 'Bloom', 'Lustre', 'Serum', 'Beauty', 'Skin', 'Face', 'Cosmetic', 'Complexion',
                     'Essence', 'Aura', 'Grace', 'Elegance', 'Purity', 'Shine', 'Polish', 'Gloss', 'Sheen', 'Sparkle',
                     'Cream', 'Lotion', 'Moisturizer', 'Toner', 'Cleanser', 'Mask', 'Scrub', 'Exfoliant', 'Peel', 'Treatment',
                     'Foundation', 'Concealer', 'Powder', 'Blush', 'Bronzer', 'Highlight', 'Contour', 'Primer', 'Setting', 'Finish',
                     'Lipstick', 'Gloss', 'Liner', 'Shadow', 'Mascara', 'Brow', 'Lash', 'Eye', 'Cheek', 'Lip',
                     'Perfume', 'Fragrance', 'Scent', 'Cologne', 'Spray', 'Mist', 'Oil', 'Balm', 'Salve', 'Butter'],
            'adjectives': [
                # TIER 2: Expanded from 40 to 75+ adjectives for beauty (1.9x increase)
                # Original core (40)
                'Radiant', 'Glowing', 'Flawless', 'Pure', 'Natural', 'Luminous', 'Vibrant', 'Fresh',
                'Dewy', 'Matte', 'Satin', 'Velvet', 'Silky', 'Smooth', 'Soft', 'Supple', 'Plump', 'Firm',
                'Youthful', 'Ageless', 'Timeless', 'Radiant', 'Bright', 'Clear', 'Even', 'Balanced', 'Healthy', 'Vital',
                'Luxury', 'Premium', 'Professional', 'Clinical', 'Derma', 'Active', 'Intensive', 'Advanced', 'Scientific', 'Bio',
                # Texture/Feel adjectives (15)
                'Creamy', 'Lightweight', 'Weightless', 'Airy', 'Whipped', 'Gel', 'Liquid', 'Powder', 'Mousse', 'Foam',
                'Rich', 'Nourishing', 'Hydrating', 'Moisturizing', 'Soothing',
                # Effect adjectives (12)
                'Brightening', 'Illuminating', 'Renewing', 'Revitalizing', 'Rejuvenating', 'Restorative', 'Repairing',
                'Protecting', 'Defending', 'Shielding', 'Lifting', 'Firming',
                # Natural/Organic adjectives (8)
                'Botanical', 'Herbal', 'Plant-Based', 'Organic', 'Clean', 'Green', 'Eco', 'Vegan'
            ],
            'themes': ['modern', 'luxury', 'natural']
        },
        'home_garden': {
            'nouns': ['Home', 'House', 'Nest', 'Haven', 'Space', 'Room', 'Place', 'Dwelling', 'Abode', 'Hearth',
                     'Garden', 'Green', 'Bloom', 'Leaf', 'Root', 'Soil', 'Plant', 'Grove', 'Yard', 'Plot'],
            'adjectives': ['Cozy', 'Warm', 'Comfortable', 'Inviting', 'Serene', 'Peaceful', 'Natural', 'Living'],
            'themes': ['natural', 'classic', 'minimal']
        },
        'jewelry': {
            'nouns': ['Gem', 'Gold', 'Silver', 'Diamond', 'Pearl', 'Crystal', 'Jewel', 'Stone', 'Ring', 'Charm',
                     'Sparkle', 'Shine', 'Lustre', 'Brilliance', 'Carat', 'Facet', 'Adorn', 'Ornament', 'Precious'],
            'adjectives': ['Precious', 'Brilliant', 'Radiant', 'Gleaming', 'Exquisite', 'Refined', 'Elegant'],
            'themes': ['luxury', 'classic']
        },
        'sports_fitness': {
            'nouns': ['Fit', 'Strength', 'Power', 'Muscle', 'Endurance', 'Performance', 'Peak', 'Force', 'Energy', 'Vigor',
                     'Athlete', 'Sport', 'Gym', 'Train', 'Workout', 'Exercise', 'Motion', 'Flex', 'Pulse', 'Pace'],
            'adjectives': ['Active', 'Strong', 'Powerful', 'Dynamic', 'Energetic', 'Athletic', 'Fierce', 'Intense'],
            'themes': ['energetic', 'modern']
        },
        'food_beverage': {
            'nouns': ['Taste', 'Flavor', 'Spice', 'Herb', 'Grain', 'Seed', 'Harvest', 'Farm', 'Kitchen', 'Pantry',
                     'Market', 'Table', 'Plate', 'Fork', 'Knife', 'Cook', 'Chef', 'Epicure', 'Gourmet', 'Delicacy'],
            'adjectives': ['Fresh', 'Savory', 'Artisan', 'Organic', 'Natural', 'Handcrafted', 'Gourmet', 'Premium'],
            'themes': ['artisan', 'natural', 'classic']
        },
        'books_media': {
            'nouns': ['Page', 'Book', 'Story', 'Chapter', 'Verse', 'Novel', 'Library', 'Read', 'Word', 'Letter',
                     'Print', 'Ink', 'Paper', 'Bound', 'Cover', 'Spine', 'Shelf', 'Volume', 'Tome', 'Text'],
            'adjectives': ['Literary', 'Classic', 'Modern', 'Timeless', 'Beloved', 'Treasured', 'Collected'],
            'themes': ['classic', 'minimal']
        },
        'pet_supplies': {
            'nouns': ['Paw', 'Tail', 'Fur', 'Whisker', 'Pet', 'Animal', 'Companion', 'Friend', 'Buddy', 'Critter',
                     'Beast', 'Creature', 'Pack', 'Den', 'Nest', 'Kennel', 'Habitat', 'Wild', 'Nature'],
            'adjectives': ['Happy', 'Healthy', 'Playful', 'Loyal', 'Friendly', 'Natural', 'Pure', 'Wild'],
            'themes': ['natural', 'modern']
        },
        'health_wellness': {
            'nouns': ['Health', 'Wellness', 'Vitality', 'Life', 'Balance', 'Harmony', 'Energy', 'Spirit', 'Body', 'Mind',
                     'Soul', 'Heal', 'Care', 'Thrive', 'Nourish', 'Restore', 'Renew', 'Revive', 'Flourish'],
            'adjectives': ['Healthy', 'Vital', 'Balanced', 'Holistic', 'Natural', 'Pure', 'Organic', 'Whole'],
            'themes': ['natural', 'modern', 'minimal']
        },
        'toys_games': {
            'nouns': ['Play', 'Toy', 'Game', 'Fun', 'Joy', 'Wonder', 'Magic', 'Dream', 'Imagine', 'Fantasy',
                     'Quest', 'Adventure', 'Discovery', 'Explore', 'Create', 'Build', 'Make', 'Puzzle', 'Brain'],
            'adjectives': ['Playful', 'Fun', 'Magical', 'Wonderful', 'Creative', 'Imaginative', 'Clever', 'Smart'],
            'themes': ['modern', 'energetic']
        }
    }
    
    # TIER 2: Regional modifiers to multiply city combinations (45 modifiers × 139 cities = 6,255 variants)
    REGIONAL_MODIFIERS = [
        # Directional (8)
        'North', 'South', 'East', 'West', 'Upper', 'Lower', 'Downtown', 'Uptown',
        # Regional descriptors (12)
        'Central', 'Metro', 'Greater', 'Historic', 'Old', 'New', 'Inner', 'Outer',
        'Urban', 'Suburban', 'Coastal', 'Highland',
        # Style modifiers (15)
        'Modern', 'Vintage', 'Classic', 'Contemporary', 'Traditional', 'Urban', 'Rustic',
        'Industrial', 'Boutique', 'Artisan', 'Craft', 'Designer', 'Luxury', 'Premium', 'Elite',
        # Spatial (10)
        'Midtown', 'Riverside', 'Lakeside', 'Hillside', 'Bayside', 'Parkside', 'Harborside',
        'Oceanside', 'Mountainside', 'Creekside'
    ]
    
    # Major cities for location-based brand names (150+ cities/regions across US/UK/Canada/Europe/Asia)
    CITIES = [
        # US Major Cities (23)
        'NYC', 'Brooklyn', 'Manhattan', 'LA', 'San Francisco', 'SF', 'Seattle', 'Portland',
        'Austin', 'Chicago', 'Boston', 'Denver', 'Miami', 'Atlanta', 'Nashville', 'Phoenix',
        'San Diego', 'Dallas', 'Houston', 'Philadelphia', 'Detroit', 'Minneapolis', 'Vegas',
        # US Mid-Tier Cities (40)
        'Tampa', 'Sacramento', 'Kansas City', 'Cleveland', 'Pittsburgh', 'Cincinnati', 'Milwaukee',
        'Indianapolis', 'Columbus', 'Charlotte', 'Raleigh', 'New Orleans', 'Salt Lake City',
        'Richmond', 'Memphis', 'Louisville', 'Buffalo', 'Tucson', 'Omaha', 'Albuquerque',
        'Tulsa', 'Fresno', 'Mesa', 'Colorado Springs', 'Arlington', 'Wichita', 'Bakersfield',
        'Aurora', 'Anaheim', 'Santa Ana', 'Corpus Christi', 'Riverside', 'Lexington', 'Stockton',
        'St. Paul', 'Newark', 'Plano', 'Henderson', 'Lincoln', 'Orlando',
        # US Coastal/Regional/Neighborhoods (20)
        'Charleston', 'Savannah', 'Santa Monica', 'Malibu', 'Beverly Hills', 'Park City',
        'Aspen', 'Napa', 'Sonoma', 'Palm Springs', 'SoHo', 'TriBeCa', 'NoMad', 'Williamsburg',
        'Silver Lake', 'Venice Beach', 'Hamptons', 'Cape Cod', 'Key West', 'Scottsdale',
        # US Regions (10)
        'SoCal', 'NorCal', 'PNW', 'Bay Area', 'Tri-State', 'New England', 'Deep South',
        'Midwest', 'Southwest', 'Pacific',
        # UK Cities (10)
        'London', 'Manchester', 'Birmingham', 'Edinburgh', 'Glasgow', 'Bristol', 'Brighton',
        'Liverpool', 'Oxford', 'Cambridge',
        # European Cities (20)
        'Paris', 'Berlin', 'Rome', 'Madrid', 'Barcelona', 'Amsterdam', 'Vienna', 'Prague',
        'Copenhagen', 'Stockholm', 'Oslo', 'Helsinki', 'Dublin', 'Brussels', 'Zurich', 'Geneva',
        'Milan', 'Florence', 'Venice', 'Lisbon',
        # Canadian Cities (6)
        'Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa', 'Victoria',
        # Asian Cities (10 - abbreviated for brand names)
        'Tokyo', 'Singapore', 'Hong Kong', 'Seoul', 'Bangkok', 'Dubai', 'Shanghai', 'Beijing',
        'Mumbai', 'Sydney'
    ]
    
    # TIER 2: Expanded business suffixes (50+ variations, 6.25x increase from original 8)
    # Directly addresses collision hotspots like "Cotton Studio", "Node Studio", "Bronzer Studio"
    SUFFIXES = {
        'company': [
            # Classic business (10)
            'Co', 'Company', 'Corporation', 'Corp', 'Inc', 'Incorporated', 'Ltd', 'Limited', 'Group', 'Enterprises',
            # Modern collective (8)
            'Collective', 'Guild', 'House', 'Works', 'Labs', 'Studio', 'Atelier', 'Maison',
            # Abbreviated (7)
            'LLC', 'Co.', 'Inc.', 'Ltd.', 'Corp.', 'Grp.', 'Ent.'
        ],
        'place': [
            # Retail spaces (12)
            'Shop', 'Store', 'Market', 'Boutique', 'Emporium', 'Gallery', 'Showroom', 'Outlet',
            'Depot', 'Warehouse', 'Bazaar', 'Mall',
            # Creative spaces (10)
            'Studio', 'Workshop', 'Atelier', 'Lab', 'Space', 'Room', 'Loft', 'Hub',
            'Factory', 'Forge'
        ],
        'descriptor': [
            # Product-focused (10)
            'Direct', 'Supply', 'Goods', 'Provisions', 'Essentials', 'Collection', 'Line', 'Series',
            'Range', 'Selection',
            # Quality-focused (8)
            'Premium', 'Select', 'Curated', 'Crafted', 'Design', 'Designs', 'Creations', 'Original'
        ],
        'familial': [
            # Family business (8)
            '& Co', '& Sons', '& Daughters', '& Brothers', '& Sisters', 'Brothers', 'Sisters', '& Family'
        ],
        'conjunction': ['&', 'and', '+', 'x', 'with']
    }
    
    # DTC-style patterns (inspired by Warby Parker, Glossier, Allbirds, etc.)
    DTC_PATTERNS = {
        'portmanteau': [  # Blend two words
            ('Ever', 'Lane'), ('War', 'by'), ('All', 'Birds'), ('Stitch', 'Fix'),
            ('Out', 'Door'), ('Nest', 'Well'), ('True', 'Classic'), ('Away', 'Travel')
        ],
        'single_word': [  # One evocative word
            'Glossier', 'Everlane', 'Reformation', 'Ritual', 'Outdoor', 'Article',
            'Floyd', 'Burrow', 'Hims', 'Hers', 'Curology', 'Function', 'Prose'
        ],
        'compound': [  # Two words combined
            'HelloFresh', 'BluApron', 'DollarShaveClub', 'ThirdLove', 'MeUndies'
        ]
    }
    
    @classmethod
    def generate_brand_name(cls, category: str, style: str = 'auto') -> str:
        """
        Generate a realistic brand name for the given category
        
        Args:
            category: Product category (fashion, electronics, beauty, etc.)
            style: Brand style (modern, luxury, artisan, dtc, etc.) or 'auto' for random
        
        Returns:
            Generated brand name string
        """
        # Normalize category
        base_category = category.replace('dtc_', '').replace('wholesale_', '')
        
        # Auto-select style if not specified
        if style == 'auto':
            if 'dtc' in category:
                # TIER 1: Maximum DTC weight to ensure hybrid patterns dominate
                style = random.choices(
                    ['dtc', 'modern', 'minimal', 'location'],
                    weights=[50, 10, 5, 35],  # Increased DTC to 50%
                    k=1
                )[0]
            elif 'wholesale' in category:
                style = random.choice(['company', 'traditional'])
            else:
                # TIER 1: Maximum DTC and location styles for high variety
                style = random.choices(
                    ['modern', 'boutique', 'descriptive', 'location', 'dtc'],
                    weights=[8, 10, 7, 30, 45],  # Increased DTC to 45%
                    k=1
                )[0]
        
        # Get category data or use generic
        category_data = cls.CATEGORY_ROOTS.get(base_category, {
            'nouns': ['Shop', 'Store', 'Market', 'Goods', 'Supply'],
            'adjectives': ['Quality', 'Premium', 'Modern', 'Fresh', 'Pure'],
            'themes': ['modern']
        })
        
        # Generate based on style
        if style == 'dtc':
            return cls._generate_dtc_name(category_data)
        elif style == 'modern':
            return cls._generate_modern_name(category_data)
        elif style == 'luxury':
            return cls._generate_luxury_name(category_data)
        elif style == 'boutique':
            return cls._generate_boutique_name(category_data)
        elif style == 'descriptive':
            return cls._generate_descriptive_name(category_data)
        elif style == 'company':
            return cls._generate_company_name(category_data)
        elif style == 'location':
            return cls._generate_location_name(category_data)
        else:
            # Random choice
            generators = [cls._generate_dtc_name, cls._generate_modern_name, 
                         cls._generate_boutique_name, cls._generate_descriptive_name,
                         cls._generate_location_name]
            return random.choice(generators)(category_data)
    
    @classmethod
    def _generate_dtc_name(cls, category_data: dict) -> str:
        """Generate DTC-style brand name (single word or portmanteau)"""
        # TIER 1 FINAL: Maximum hybrid pattern weight for 250K scale uniqueness
        # Hybrid patterns combine high-variety elements for maximum uniqueness
        pattern = random.choices(
            ['portmanteau', 'single_word', 'compound', 'numeric', 'initial', 'location', 'hybrid'],
            weights=[2, 1, 1, 26, 20, 23, 27],  # Hybrid at 27%, highest of all patterns
            k=1
        )[0]
        
        if pattern == 'portmanteau':
            # Blend two nouns or adj + noun
            if random.random() < 0.5:
                word1 = random.choice(category_data['nouns'])[:4]
                word2 = random.choice(category_data['nouns'])[2:]
            else:
                word1 = random.choice(category_data['adjectives'])[:4]
                word2 = random.choice(category_data['nouns'])
            return f"{word1}{word2}"
        
        elif pattern == 'single_word':
            # TIER 2: Single evocative word with optional descriptive prefix
            # Examples: "Glossier", "Premium Silk", "Vintage Fashion"
            word = random.choice(category_data['nouns'] + category_data['adjectives'])
            
            # 20% chance to add descriptive prefix
            if random.random() < 0.20:
                prefix = random.choice(cls.DESCRIPTIVE_PREFIXES)
                return f"{prefix} {word}"
            else:
                return word
        
        elif pattern == 'numeric':
            # TIER 1: Strategic numeric patterns (years, zip codes, phone-style, random)
            word = random.choice(category_data['nouns'])
            
            # Choose numeric type strategically
            numeric_type = random.choices(
                ['random', 'year', 'zip', 'phone', 'suffix'],
                weights=[40, 20, 15, 15, 10],  # Balanced mix
                k=1
            )[0]
            
            if numeric_type == 'year':
                year = random.randint(1900, 2025)
                formats = [f"Est. {year}", f"Since {year}", f"{word} {year}", f"{word} '{year % 100:02d}"]
                return random.choice(formats)
            elif numeric_type == 'zip':
                zip_code = random.randint(10000, 99999)
                formats = [f"{word} {zip_code}", f"{word}{zip_code}"]
                return random.choice(formats)
            elif numeric_type == 'phone':
                phone = random.randint(1000, 9999)  # Last 4 digits style
                formats = [f"{word} {phone}", f"{word}-{phone}"]
                return random.choice(formats)
            elif numeric_type == 'suffix':
                suffix_word = random.choice(['Co', 'Lab', 'Studio'])
                return f"{word} {suffix_word}"
            else:  # random
                num = random.randint(1, 9999)
                # For large numbers, prefer formats without ordinals
                if num > 100:
                    formats = [f"{word} {num}", f"{word}{num}", f"No. {num} {word}", f"{word} No. {num}"]
                else:
                    formats = [f"{word} {num}", f"The {num}th {word}", f"{word}{num}", f"No. {num} {word}"]
                return random.choice(formats)
        
        elif pattern == 'initial':
            # TIER 1: Support 2, 3, and 4-letter initials (AAAA-ZZZZ = 456K combinations)
            import string
            word = random.choice(category_data['nouns'])
            
            # Choose initial length: 60% 2-letter, 30% 3-letter, 10% 4-letter
            initial_type = random.choices([2, 3, 4], weights=[60, 30, 10], k=1)[0]
            
            if initial_type == 4:
                # 4-letter initials (456,976 combinations)
                initials = ''.join(random.choices(string.ascii_uppercase, k=4))
                formats = [
                    f"{initials} {word}",
                    f"{'.'.join(initials)}. {word}",
                    f"{initials}"
                ]
                return random.choice(formats)
            elif initial_type == 3:
                # 3-letter initials (17,576 combinations)
                initials = ''.join(random.choices(string.ascii_uppercase, k=3))
                formats = [
                    f"{initials} {word}",
                    f"{'.'.join(initials)}. {word}",
                    f"{initials}"
                ]
                return random.choice(formats)
            else:
                # 2-letter initials (676 combinations)
                initials = ''.join(random.choices(string.ascii_uppercase, k=2))
                formats = [
                    f"{initials} {word}",
                    f"{initials[0]}&{initials[1]} {word}",
                    f"{'.'.join(initials)}. {word}"
                ]
                return random.choice(formats)
        
        elif pattern == 'location':
            # TIER 2: City-based brands with optional regional modifiers
            # Examples: "North Seattle Lens", "Downtown Brooklyn 247", "Modern Paris Fashion"
            city = random.choice(cls.CITIES)
            word = random.choice(category_data['nouns'])
            
            # 30% chance to add regional modifier for uniqueness boost
            use_modifier = random.random() < 0.30
            if use_modifier:
                modifier = random.choice(cls.REGIONAL_MODIFIERS)
                # Modifier + City patterns
                formats = [
                    f"{modifier} {city} {word}",
                    f"{modifier} {city} {word} Co",
                    f"{modifier} {city} {word} Studio",
                    f"{word} of {modifier} {city}",
                    f"{modifier} {city}'s {word}",
                    f"The {modifier} {city} {word}"
                ]
            else:
                # Original formats without modifier
                formats = [
                    f"{city} {word}",
                    f"{word} {city}",
                    f"The {city} {word}",
                    f"{city} {word} Co",
                    f"{city} {word} Studio",
                    f"{city} {word} Shop",
                    f"{city} {word} Market",
                    f"{word} of {city}",
                    f"{city}'s {word}",
                    f"{city} {word} House",
                    f"{city} {word} Collective",
                    f"{word} {city} Co"
                ]
            return random.choice(formats)
        
        elif pattern == 'hybrid':
            # TIER 1 + TIER 2: Hybrid patterns combine high-variety elements + regional modifiers
            # Types: City+Number, Initial+City, Number+Location, Triple hybrid
            import string
            
            hybrid_type = random.choices(
                ['city_number', 'initial_city', 'number_location', 'triple'],
                weights=[35, 35, 20, 10],
                k=1
            )[0]
            
            city = random.choice(cls.CITIES)
            word = random.choice(category_data['nouns'])
            
            if hybrid_type == 'city_number':
                # TIER 2: City + Number with optional regional modifier
                # Examples: "Brooklyn 247", "North LA 1842", "Downtown Tokyo No. 5"
                num = random.randint(1, 9999)
                
                # 25% chance for regional modifier (lower than pure location patterns)
                use_modifier = random.random() < 0.25
                if use_modifier:
                    modifier = random.choice(cls.REGIONAL_MODIFIERS)
                    formats = [
                        f"{modifier} {city} {num}",
                        f"{modifier} {city} No. {num}",
                        f"No. {num} {modifier} {city}"
                    ]
                else:
                    formats = [
                        f"{city} {num}",
                        f"{city} {word} {num}",
                        f"{word} {city} {num}",
                        f"{city} No. {num}"
                    ]
                return random.choice(formats)
            
            elif hybrid_type == 'initial_city':
                # TIER 2: Initial + City with optional regional modifier
                # Examples: "AB Fashion NYC", "XYZ Modern LA", "MK Downtown Tokyo"
                initials = ''.join(random.choices(string.ascii_uppercase, k=random.choices([2, 3], weights=[70, 30], k=1)[0]))
                
                # 25% chance for regional modifier
                use_modifier = random.random() < 0.25
                if use_modifier:
                    modifier = random.choice(cls.REGIONAL_MODIFIERS)
                    formats = [
                        f"{initials} {modifier} {city}",
                        f"{initials} {word} {modifier} {city}",
                        f"{modifier} {city} {initials}"
                    ]
                else:
                    formats = [
                        f"{initials} {word} {city}",
                        f"{initials} {city} {word}",
                        f"{city} {initials} {word}",
                        f"{initials} {city}"
                    ]
                return random.choice(formats)
            
            elif hybrid_type == 'number_location':
                # Number + Location: "247 Brooklyn Fashion", "Studio 42 NYC"
                num = random.randint(1, 999)
                formats = [
                    f"{num} {city} {word}",
                    f"{word} {num} {city}",
                    f"No. {num} {city}"
                ]
                return random.choice(formats)
            
            else:  # triple
                # Triple hybrid: "AB Brooklyn 247", "XYZ LA 1842"
                initials = ''.join(random.choices(string.ascii_uppercase, k=2))
                num = random.randint(1, 999)
                formats = [
                    f"{initials} {city} {num}",
                    f"{initials} {num} {city}",
                    f"{city} {initials} {num}"
                ]
                return random.choice(formats)
        
        else:  # compound
            # TIER 2: Compound pattern with optional descriptive prefix
            # Examples: "HelloFresh", "Premium SilkStyle", "Vintage FashionWear"
            word1 = random.choice(category_data['adjectives'])
            word2 = random.choice(category_data['nouns'])
            compound = f"{word1}{word2}"
            
            # 20% chance to add descriptive prefix
            if random.random() < 0.20:
                prefix = random.choice(cls.DESCRIPTIVE_PREFIXES)
                return f"{prefix} {compound}"
            else:
                return compound
    
    @classmethod
    def _generate_modern_name(cls, category_data: dict) -> str:
        """Generate modern brand name (Prefix + Noun + optional suffix)"""
        theme = random.choice(category_data.get('themes', ['modern']))
        prefix = random.choice(cls.PREFIXES[theme])
        noun = random.choice(category_data['nouns'])
        
        if random.random() < 0.3:
            suffix = random.choice(cls.SUFFIXES['company'])
            return f"{prefix} {noun} {suffix}"
        else:
            return f"{prefix} {noun}"
    
    @classmethod
    def _generate_luxury_name(cls, category_data: dict) -> str:
        """Generate luxury brand name"""
        prefix = random.choice(cls.PREFIXES['luxury'])
        noun = random.choice(category_data['nouns'])
        
        patterns = [
            f"{prefix} {noun}",
            f"{noun} {prefix}",
            f"The {prefix} {noun}",
            f"{prefix} & {noun}"
        ]
        return random.choice(patterns)
    
    @classmethod
    def _generate_boutique_name(cls, category_data: dict) -> str:
        """Generate boutique-style name"""
        adj = random.choice(category_data['adjectives'])
        noun = random.choice(category_data['nouns'])
        place = random.choice(cls.SUFFIXES['place'])
        
        patterns = [
            f"The {adj} {noun}",
            f"{adj} {noun} {place}",
            f"{noun} {place}",
            f"The {noun} {place}"
        ]
        return random.choice(patterns)
    
    @classmethod
    def _generate_descriptive_name(cls, category_data: dict) -> str:
        """Generate descriptive brand name (Adj + Noun + Descriptor)"""
        adj = random.choice(category_data['adjectives'])
        noun = random.choice(category_data['nouns'])
        descriptor = random.choice(cls.SUFFIXES['descriptor'])
        
        patterns = [
            f"{adj} {noun}",
            f"{adj} {noun} {descriptor}",
            f"{noun} {descriptor}",
            f"{adj} {descriptor}"
        ]
        return random.choice(patterns)
    
    @classmethod
    def _generate_company_name(cls, category_data: dict) -> str:
        """Generate wholesale/B2B company name"""
        noun = random.choice(category_data['nouns'])
        suffix = random.choice(cls.SUFFIXES['company'])
        
        patterns = [
            f"{noun} {suffix}",
            f"National {noun} {suffix}",
            f"Premier {noun} {suffix}",
            f"Metro {noun} {suffix}",
            f"{noun} Wholesale {suffix}",
            f"{noun} Distributors {suffix}"
        ]
        return random.choice(patterns)
    
    @classmethod
    def _generate_location_name(cls, category_data: dict) -> str:
        """Generate location-based brand name"""
        city = random.choice(cls.CITIES)
        noun = random.choice(category_data['nouns'])
        adj = random.choice(category_data['adjectives'])
        
        # Expanded format variety (14 patterns)
        patterns = [
            f"{city} {noun}",
            f"{noun} {city}",
            f"The {city} {noun}",
            f"{city} {adj} {noun}",
            f"{adj} {city} {noun}",
            f"{city} {noun} Co",
            f"{city} {noun} Studio",
            f"{city} {noun} Shop",
            f"{city} {noun} Market",
            f"{noun} of {city}",
            f"{city}'s {noun}",
            f"{city} {noun} House",
            f"{city} {noun} Collective",
            f"{adj} {noun} {city}"
        ]
        return random.choice(patterns)
    
    @classmethod
    def generate_multiple(cls, category: str, count: int = 10) -> List[str]:
        """Generate multiple unique brand names for a category"""
        names = set()
        attempts = 0
        max_attempts = count * 10
        
        while len(names) < count and attempts < max_attempts:
            name = cls.generate_brand_name(category)
            names.add(name)
            attempts += 1
        
        return list(names)
