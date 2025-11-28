"""
Retail-Specific Data Generator
Generates complete POS receipt and e-commerce order data with ALL 37 retail entities
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import random
from faker import Faker


@dataclass
class RetailLineItem:
    """Retail line item with complete entity coverage"""
    description: str
    quantity: int
    unit_price: float
    total: float
    upc: str  # ITEM_SKU
    sku: Optional[str] = None  # ITEM_SKU alternative
    unit: str = "ea"  # ITEM_UNIT
    tax_rate: float = 0.0
    tax_amount: float = 0.0  # ITEM_TAX
    discount: float = 0.0  # ITEM_DISCOUNT
    lot_number: Optional[str] = None  # LOT_NUMBER
    serial_number: Optional[str] = None  # SERIAL_NUMBER
    weight: Optional[float] = None  # WEIGHT
    promotion: Optional[str] = None
    rewards_earned: Optional[int] = None


@dataclass
class RetailReceiptData:
    """Complete retail receipt data with all 37 entities"""
    
    # Document metadata (4 entities)
    doc_type: str = "Receipt"  # DOC_TYPE
    invoice_number: str = ""  # INVOICE_NUMBER
    invoice_date: str = ""  # INVOICE_DATE
    order_date: Optional[str] = None  # ORDER_DATE
    
    # Merchant information (4 entities)
    supplier_name: str = ""  # SUPPLIER_NAME
    supplier_address: str = ""  # SUPPLIER_ADDRESS
    supplier_phone: str = ""  # SUPPLIER_PHONE
    supplier_email: str = ""  # SUPPLIER_EMAIL
    
    # Customer information (4 entities) - optional for POS
    buyer_name: Optional[str] = None  # BUYER_NAME
    buyer_address: Optional[str] = None  # BUYER_ADDRESS
    buyer_phone: Optional[str] = None  # BUYER_PHONE
    buyer_email: Optional[str] = None  # BUYER_EMAIL
    
    # Financial totals (8 entities)
    currency: str = "$"  # CURRENCY
    subtotal: float = 0.0  # SUBTOTAL
    tax_amount: float = 0.0  # TAX_AMOUNT
    tax_rate: float = 0.0  # TAX_RATE
    total_amount: float = 0.0  # TOTAL_AMOUNT
    discount: float = 0.0  # DISCOUNT
    payment_method: str = "Cash"  # PAYMENT_METHOD
    payment_terms: Optional[str] = None  # PAYMENT_TERMS (card details, approval codes)
    
    # Line items (9 entities - handled in RetailLineItem)
    line_items: List[RetailLineItem] = field(default_factory=list)
    
    # Retail identifiers (4 entities)
    register_number: Optional[str] = None  # REGISTER_NUMBER
    cashier_id: Optional[str] = None  # CASHIER_ID
    tracking_number: Optional[str] = None  # TRACKING_NUMBER (for online orders)
    account_number: Optional[str] = None  # ACCOUNT_NUMBER (loyalty/member ID)
    
    # Miscellaneous (3 entities)
    terms_and_conditions: Optional[str] = None  # TERMS_AND_CONDITIONS
    note: Optional[str] = None  # NOTE
    
    # Structural (1 entity)
    has_table: bool = True  # TABLE (line items table)
    
    # Additional template fields (not in label schema but needed for templates)
    transaction_time: Optional[str] = None
    transaction_number: Optional[str] = None
    store_website: Optional[str] = None
    customer_id: Optional[str] = None
    total_discount: float = 0.0
    tip_amount: float = 0.0
    card_type: Optional[str] = None
    card_last_four: Optional[str] = None
    approval_code: Optional[str] = None
    transaction_id: Optional[str] = None
    cash_tendered: Optional[float] = None
    change_amount: Optional[float] = None
    return_policy: str = "Returns accepted within 30 days with receipt"
    footer_message: str = "Have a great day!"
    
    # Loyalty/rewards
    loyalty_points_earned: Optional[int] = None
    loyalty_points_balance: Optional[int] = None
    loyalty_rewards_available: Optional[int] = None
    
    # Survey
    survey_url: Optional[str] = None
    survey_code: Optional[str] = None
    
    # Barcode
    barcode_value: Optional[str] = None
    barcode_image: Optional[str] = None
    
    # Locale for formatting
    locale: str = 'en_US'


class RetailDataGenerator:
    """Generates retail-specific synthetic data ensuring ALL 37 entities appear"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        self.fake = Faker(locale)
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # Retail product categories (optimized for Shopify use cases)
        # Expanded to 50+ items per category for production readiness
        self.fashion_items = [
            # Tops
            "Cotton T-Shirt", "V-Neck Tee", "Tank Top", "Crop Top", "Blouse",
            "Button-Down Shirt", "Polo Shirt", "Henley Shirt", "Tunic Top",
            "Sweater", "Cardigan", "Hoodie", "Pullover Sweatshirt",
            # Bottoms
            "Denim Jeans", "Skinny Jeans", "Wide Leg Pants", "Yoga Pants", "Leggings",
            "Cargo Pants", "Chinos", "Shorts", "Denim Shorts", "Maxi Skirt",
            "Mini Skirt", "Pencil Skirt", "Pleated Skirt",
            # Dresses & Jumpsuits
            "Floral Dress", "Maxi Dress", "Midi Dress", "Wrap Dress", "Shirt Dress",
            "Jumpsuit", "Romper",
            # Outerwear
            "Leather Jacket", "Denim Jacket", "Blazer", "Trench Coat", "Parka",
            "Windbreaker", "Bomber Jacket", "Puffer Vest",
            # Activewear
            "Sports Bra", "Athletic Shorts", "Running Tights", "Track Pants",
            "Performance Tank", "Training Hoodie",
            # Sleepwear & Loungewear
            "Pajama Set", "Nightgown", "Lounge Pants", "Sleep Shirt", "Robe"
        ]
        
        self.accessories_items = [
            # Bags
            "Leather Handbag", "Canvas Tote Bag", "Crossbody Bag", "Backpack",
            "Messenger Bag", "Clutch Purse", "Weekender Bag", "Laptop Bag",
            "Beach Bag", "Diaper Bag", "Fanny Pack", "Drawstring Bag",
            # Hats & Headwear
            "Baseball Cap", "Beanie", "Fedora", "Wide Brim Hat", "Bucket Hat",
            "Sun Hat", "Beret", "Trucker Hat",
            # Scarves & Wraps
            "Silk Scarf", "Infinity Scarf", "Blanket Scarf", "Pashmina Shawl",
            # Belts & Wallets
            "Leather Belt", "Canvas Belt", "Chain Belt", "Leather Wallet",
            "Card Holder", "Money Clip", "Coin Purse",
            # Eyewear
            "Sunglasses", "Reading Glasses", "Blue Light Glasses", "Sports Sunglasses",
            # Hair Accessories
            "Hair Clips Set", "Headband", "Scrunchie Set", "Hair Ties Pack",
            "Bobby Pins", "Barrettes",
            # Watches
            "Stainless Steel Watch", "Leather Watch", "Smart Watch", "Sport Watch",
            # Other
            "Umbrella", "Keychain", "Phone Charm", "Luggage Tag", "Bandana"
        ]
        
        self.jewelry_items = [
            # Necklaces
            "Sterling Silver Necklace", "Gold Chain Necklace", "Rose Gold Pendant",
            "Layered Necklace", "Choker Necklace", "Pearl Necklace", "Locket Necklace",
            "Bar Necklace", "Name Necklace", "Cross Necklace",
            # Earrings
            "Gold Hoop Earrings", "Gemstone Stud Earrings", "Drop Earrings",
            "Dangle Earrings", "Huggie Hoops", "Pearl Studs", "Crystal Earrings",
            "Threader Earrings", "Ear Cuffs",
            # Bracelets
            "Pearl Bracelet", "Charm Bracelet", "Cuff Bracelet", "Bangle Set",
            "Tennis Bracelet", "Chain Bracelet", "Leather Bracelet", "Beaded Bracelet",
            "Friendship Bracelet", "Anklet",
            # Rings
            "Diamond Ring", "Engagement Ring", "Wedding Band", "Stackable Rings",
            "Statement Ring", "Midi Ring", "Signet Ring", "Birthstone Ring",
            "Promise Ring", "Eternity Band",
            # Sets
            "Jewelry Set", "Bridal Set", "Necklace & Earring Set",
            # Body Jewelry
            "Belly Ring", "Nose Ring", "Toe Ring", "Body Chain",
            # Other
            "Brooch Pin", "Lapel Pin", "Cufflinks", "Tie Clip"
        ]
        
        self.beauty_items = [
            # Skincare
            "Organic Face Serum", "Hydrating Moisturizer", "Vitamin C Serum",
            "Retinol Cream", "Hyaluronic Acid Serum", "Facial Cleanser", "Toner",
            "Face Mask Set", "Sheet Mask", "Eye Cream", "Night Cream", "Day Cream",
            "Sunscreen SPF 50", "Exfoliating Scrub", "Micellar Water", "Essence",
            # Makeup - Face
            "BB Cream", "CC Cream", "Foundation", "Concealer", "Powder",
            "Blush", "Bronzer", "Highlighter", "Setting Spray", "Primer",
            # Makeup - Eyes
            "Eye Shadow Palette", "Mascara", "Eyeliner", "Eyebrow Pencil",
            "Brow Gel", "Eye Primer", "False Lashes",
            # Makeup - Lips
            "Matte Lipstick", "Lip Gloss", "Lip Liner", "Liquid Lipstick",
            "Lip Balm", "Lip Stain", "Lip Scrub",
            # Tools & Brushes
            "Makeup Brush Set", "Beauty Blender", "Eyelash Curler", "Tweezers",
            "Makeup Sponge", "Brush Cleaner",
            # Nails
            "Nail Polish", "Gel Polish", "Nail File Set", "Cuticle Oil",
            # Haircare
            "Hair Styling Cream", "Hair Mask", "Dry Shampoo", "Hair Oil",
            "Leave-In Conditioner",
            # Fragrance
            "Perfume 50ml", "Body Spray", "Cologne", "Roll-On Perfume"
        ]
        
        self.home_garden_items = [
            # Decor
            "Throw Pillow", "Decorative Pillow", "Cushion Cover", "Throw Blanket",
            "Wall Art Print", "Canvas Wall Art", "Metal Wall Art", "Tapestry",
            "Decorative Mirror", "Picture Frame", "Photo Collage Frame",
            "Scented Candle", "Candle Holder", "Tea Light Set", "Diffuser",
            "Ceramic Vase", "Glass Vase", "Artificial Plant", "Succulent Set",
            "String Lights", "Fairy Lights", "LED Strip Lights", "Table Lamp",
            "Floor Lamp", "Desk Lamp", "Night Light",
            # Textiles
            "Bath Towel Set", "Hand Towel", "Washcloth Set", "Bath Mat",
            "Shower Curtain", "Table Runner", "Placemats Set", "Napkin Set",
            "Area Rug", "Door Mat", "Kitchen Towel Set",
            # Storage & Organization
            "Storage Basket", "Storage Bin", "Shelf Organizer", "Drawer Divider",
            "Jewelry Box", "Makeup Organizer", "Closet Organizer",
            # Garden
            "Plant Pot", "Planter Box", "Watering Can", "Garden Tools Set",
            "Plant Stand", "Hanging Planter", "Seed Starter Kit", "Garden Gloves"
        ]
        
        self.sports_fitness_items = [
            # Equipment
            "Yoga Mat", "Exercise Mat", "Foam Roller", "Resistance Bands",
            "Resistance Loop Bands", "Jump Rope", "Exercise Ball", "Medicine Ball",
            "Dumbbells 10lb", "Dumbbells 20lb", "Kettlebell", "Weight Set",
            "Pull Up Bar", "Push Up Bars", "Ab Wheel", "Balance Ball",
            "Yoga Blocks", "Yoga Strap", "Massage Ball",
            # Apparel
            "Gym Bag", "Duffle Bag", "Workout Gloves", "Lifting Straps",
            "Compression Socks", "Compression Sleeves", "Sweatband Set",
            "Athletic Headband", "Workout Towel",
            # Nutrition
            "Protein Powder", "Protein Bar Box", "Pre-Workout", "BCAA Powder",
            "Creatine", "Protein Shaker", "Pill Organizer", "Supplement Container",
            # Hydration
            "Water Bottle", "Insulated Water Bottle", "Sport Water Bottle",
            "Hydration Pack", "Electrolyte Powder",
            # Tech & Accessories
            "Fitness Tracker", "Heart Rate Monitor", "Stopwatch", "Pedometer",
            "Armband Phone Holder", "Workout Timer", "Bluetooth Earbuds",
            # Recovery
            "Ice Pack", "Heating Pad", "Massage Gun", "Kinesiology Tape"
        ]
        
        self.pet_supplies_items = [
            # Food & Treats
            "Dog Food", "Cat Food", "Dog Treats", "Cat Treats", "Dental Chews",
            "Bully Sticks", "Rawhide Bones", "Freeze-Dried Treats",
            # Bowls & Feeders
            "Food Bowl Set", "Water Bowl", "Elevated Feeder", "Automatic Feeder",
            "Slow Feeder Bowl", "Travel Bowl",
            # Toys
            "Cat Toy", "Dog Toy", "Chew Toys", "Rope Toy", "Plush Toy",
            "Interactive Toy", "Puzzle Toy", "Ball Launcher", "Catnip Toys",
            "Feather Wand", "Laser Pointer",
            # Beds & Furniture
            "Pet Bed", "Dog Bed", "Cat Bed", "Pet Blanket", "Pet Mat",
            "Cat Tree", "Scratching Post", "Pet Stairs", "Pet Ramp",
            # Leashes & Collars
            "Dog Collar", "Cat Collar", "Leash", "Retractable Leash", "Harness",
            "ID Tag", "Collar Charm",
            # Grooming
            "Pet Shampoo", "Grooming Brush", "Nail Clippers", "Pet Wipes",
            "Deshedding Tool", "Toothbrush Set", "Ear Cleaner",
            # Litter & Cleanup
            "Cat Litter", "Litter Box", "Litter Scoop", "Waste Bags", "Pet Odor Spray",
            # Travel & Carriers
            "Pet Carrier", "Travel Crate", "Car Seat Cover", "Pet Seatbelt"
        ]
        
        self.books_media_items = [
            # Books
            "Hardcover Book", "Paperback Book", "Comic Book", "Graphic Novel",
            "Coloring Book", "Activity Book", "Cookbook", "Self-Help Book",
            "Children's Book", "Young Adult Novel", "Fantasy Novel", "Mystery Book",
            # Journals & Notebooks
            "Journal", "Bullet Journal", "Travel Journal", "Gratitude Journal",
            "Notebook", "Sketchbook", "Composition Notebook", "Spiral Notebook",
            "Planner", "Daily Planner", "Weekly Planner", "Agenda",
            # Art & Prints
            "Art Print", "Poster", "Canvas Print", "Framed Print",
            "Photography Print", "Vintage Poster",
            # Stationery
            "Pen Set", "Pencil Set", "Marker Set", "Highlighter Set",
            "Sticky Notes", "Washi Tape", "Sticker Pack", "Bookmark Set",
            "Greeting Cards", "Thank You Cards", "Note Cards",
            # Accessories
            "Book Light", "Reading Light", "Book Stand", "Bookends",
            "Book Sleeve", "Page Markers",
            # Subscriptions & Media
            "Magazine Subscription", "Digital Download", "E-Book", "Audiobook"
        ]
        
        self.toys_games_items = [
            # Puzzles
            "Puzzle 1000pc", "Puzzle 500pc", "3D Puzzle", "Jigsaw Puzzle",
            "Brain Teaser", "Puzzle Mat",
            # Board Games & Cards
            "Board Game", "Strategy Game", "Party Game", "Family Game",
            "Card Game", "Playing Cards", "Trivia Game", "Chess Set",
            "Checkers", "Backgammon", "Dice Set",
            # Action Figures & Dolls
            "Action Figure", "Doll", "Fashion Doll", "Baby Doll",
            "Doll House", "Action Figure Set", "Collectible Figure",
            # Building & Construction
            "Building Blocks", "LEGO Set", "Construction Set", "Marble Run",
            "Magnetic Tiles", "Wooden Blocks",
            # Stuffed Animals
            "Stuffed Animal", "Plush Toy", "Teddy Bear", "Unicorn Plush",
            # Remote Control & Tech
            "Remote Control Car", "RC Truck", "Drone", "Robot Toy",
            # Arts & Crafts
            "Art Supplies", "Craft Kit", "Paint Set", "Crayon Set",
            "Play-Doh Set", "Modeling Clay", "Bead Kit", "Sewing Kit",
            "Origami Paper", "Coloring Set", "Watercolor Set",
            # Outdoor & Active
            "Toy Car", "Toy Train", "Toy Kitchen", "Play Tent",
            "Water Toys", "Sand Toys", "Bubble Machine", "Kite",
            # Educational
            "STEM Kit", "Science Kit", "Learning Toy", "Flash Cards"
        ]
        
        self.food_beverage_items = [
            # Coffee & Tea
            "Organic Coffee Beans", "Ground Coffee", "Instant Coffee", "Cold Brew",
            "Specialty Tea", "Green Tea", "Herbal Tea", "Black Tea",
            "Tea Sampler", "Matcha Powder", "Chai Tea", "Kombucha",
            # Chocolate & Sweets
            "Artisan Chocolate", "Dark Chocolate Bar", "Milk Chocolate", "Truffles",
            "Chocolate Gift Box", "Fudge", "Caramels", "Gourmet Candy",
            # Oils & Condiments
            "Gourmet Olive Oil", "Avocado Oil", "Coconut Oil", "Truffle Oil",
            "Balsamic Vinegar", "Apple Cider Vinegar", "Craft Hot Sauce",
            "BBQ Sauce", "Salsa", "Mustard", "Pesto", "Jam Preserves",
            # Snacks
            "Protein Bar Box", "Energy Bars", "Granola Bars", "Trail Mix",
            "Premium Nuts", "Roasted Almonds", "Cashews", "Mixed Nuts",
            "Dried Fruit", "Dried Mango", "Fruit Chips", "Veggie Chips",
            "Popcorn", "Rice Cakes", "Crackers", "Pretzels",
            # Sweeteners & Spreads
            "Raw Honey", "Maple Syrup", "Agave Nectar", "Nut Butter",
            "Almond Butter", "Peanut Butter", "Tahini",
            # Other
            "Organic Granola", "Protein Powder", "Spice Set", "Seasoning Blend",
            "Vanilla Extract", "Baking Mix"
        ]
        
        self.health_wellness_items = [
            # Vitamins
            "Multivitamin", "Vitamin D3", "Vitamin C", "Vitamin B Complex",
            "Vitamin E", "Vitamin K2", "Prenatal Vitamins", "Men's Multivitamin",
            "Women's Multivitamin", "Kids Vitamins",
            # Minerals
            "Magnesium", "Calcium", "Zinc", "Iron Supplement", "Potassium",
            # Supplements
            "Omega-3 Fish Oil", "Probiotics", "Collagen Powder", "Biotin",
            "Turmeric Supplement", "Ashwagandha", "Glucosamine", "CoQ10",
            "Apple Cider Vinegar Pills", "Green Tea Extract", "Spirulina",
            "Fiber Supplement", "Digestive Enzymes",
            # Protein & Fitness
            "Protein Shake", "Protein Powder", "Pre-Workout", "Creatine",
            "BCAA", "Mass Gainer", "Meal Replacement Shake",
            # Sleep & Relaxation
            "Melatonin", "Valerian Root", "Sleep Aid", "Stress Relief",
            # CBD & Natural
            "CBD Oil", "CBD Gummies", "Hemp Oil", "Essential Oil Set",
            "Lavender Oil", "Peppermint Oil", "Tea Tree Oil",
            # Herbal Teas & Drinks
            "Herbal Tea", "Detox Tea", "Slim Tea", "Energy Drink Mix",
            # Other
            "Electrolyte Powder", "Immune Support", "Joint Support", "Hair Growth"
        ]
        
        self.electronics_items = [
            # Cables & Chargers
            "USB-C Cable", "Lightning Cable", "Micro USB Cable", "HDMI Cable",
            "Aux Cable", "Ethernet Cable", "Extension Cord", "Power Strip",
            "Portable Charger", "Wall Charger", "Car Charger", "Wireless Charger",
            "Charging Station", "USB Hub", "Cable Organizer",
            # Audio
            "Wireless Earbuds", "Wired Earbuds", "Bluetooth Headphones",
            "Over-Ear Headphones", "Gaming Headset", "Bluetooth Speaker",
            "Portable Speaker", "Microphone",
            # Phone Accessories
            "Phone Case", "Clear Phone Case", "Wallet Phone Case", "Phone Grip",
            "Screen Protector", "Tempered Glass", "Pop Socket", "Phone Stand",
            "Car Phone Mount", "Phone Ring Holder", "Armband Phone Holder",
            # Computer & Laptop
            "Laptop Sleeve", "Laptop Stand", "Laptop Bag", "Mouse Pad",
            "Wireless Mouse", "Wired Mouse", "Keyboard", "Webcam",
            "USB Flash Drive", "External Hard Drive", "SD Card", "Card Reader",
            # Smart Watch Accessories
            "Smart Watch Band", "Watch Screen Protector", "Watch Charging Cable",
            # Camera & Photography
            "Ring Light", "LED Light", "Selfie Stick", "Tripod", "Phone Lens Kit",
            # Other
            "Stylus Pen", "Tablet Case", "Tablet Stand", "Cable Clips"
        ]
        
        # Payment methods (PAYMENT_METHOD)
        self.payment_methods = [
            "Visa", "Mastercard", "American Express", "Discover",
            "Debit Card", "Cash", "Gift Card", "Apple Pay", "Google Pay"
        ]
        
        # Store types (optimized for Shopify merchants)
        self.store_types = {
            'fashion': ['Trendy Threads', 'Style Boutique', 'Modern Wardrobe', 'Chic Apparel'],
            'accessories': ['Accessory Bar', 'The Handbag Shop', 'Urban Accessories', 'Style Co'],
            'jewelry': ['Gem & Gold', 'Sparkle Jewelry', 'Luxe Jewelers', 'Diamond District'],
            'beauty': ['Glow Beauty', 'Pure Cosmetics', 'Beauty Haven', 'Radiance Shop'],
            'home_garden': ['Home Style', 'Garden & Decor', 'Cozy Living', 'Interior Luxe'],
            'sports_fitness': ['Fit Life', 'Active Gear', 'Fitness Hub', 'Sport Zone'],
            'pet_supplies': ['Pet Paradise', 'Pawfect Pets', 'Animal Care', 'Furry Friends'],
            'books_media': ['Book Nook', 'Page Turner', 'The Reading Room', 'Literary Corner'],
            'toys_games': ['Play Time', 'Toy Box', 'Game Central', 'Kids Paradise'],
            'food_beverage': ['Artisan Foods', 'Gourmet Market', 'Specialty Eats', 'Fresh & Pure'],
            'health_wellness': ['Wellness Shop', 'Healthy Living', 'Vitality Store', 'Pure Health'],
            'electronics': ['Tech Essentials', 'Gadget Box', 'Digital Life', 'Mobile Gear']
        }
    
    def generate_line_item(self, category: str = 'fashion') -> RetailLineItem:
        """Generate a single line item with complete entity coverage"""
        
        # Select product based on category (Shopify-focused)
        if category == 'fashion':
            description = random.choice(self.fashion_items)
            unit = 'ea'
        elif category == 'accessories':
            description = random.choice(self.accessories_items)
            unit = 'ea'
        elif category == 'jewelry':
            description = random.choice(self.jewelry_items)
            unit = 'ea'
        elif category == 'beauty':
            description = random.choice(self.beauty_items)
            unit = random.choice(['ea', 'ml', 'oz'])
        elif category == 'home_garden':
            description = random.choice(self.home_garden_items)
            unit = 'ea'
        elif category == 'sports_fitness':
            description = random.choice(self.sports_fitness_items)
            unit = random.choice(['ea', 'lb', 'oz'])
        elif category == 'pet_supplies':
            description = random.choice(self.pet_supplies_items)
            unit = random.choice(['ea', 'lb', 'oz'])
        elif category == 'books_media':
            description = random.choice(self.books_media_items)
            unit = 'ea'
        elif category == 'toys_games':
            description = random.choice(self.toys_games_items)
            unit = 'ea'
        elif category == 'food_beverage':
            description = random.choice(self.food_beverage_items)
            unit = random.choice(['ea', 'lb', 'oz', 'pkg'])
        elif category == 'health_wellness':
            description = random.choice(self.health_wellness_items)
            unit = random.choice(['ea', 'capsules', 'oz'])
        elif category == 'electronics':
            description = random.choice(self.electronics_items)
            unit = 'ea'
        else:
            description = self.fake.word().title() + " Product"
            unit = 'ea'
        
        # ITEM_QTY, ITEM_UNIT_COST, ITEM_TOTAL_COST
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(1.99, 49.99), 2)
        total = round(quantity * unit_price, 2)
        
        # ITEM_SKU (UPC)
        upc = self.fake.ean13()
        sku = self.fake.bothify(text='SKU-######')
        
        # ITEM_TAX
        tax_rate = random.choice([0.0, 6.5, 7.5, 8.25, 9.0])
        tax_amount = round(total * (tax_rate / 100), 2) if tax_rate > 0 else 0.0
        
        # ITEM_DISCOUNT (30% chance)
        discount = 0.0
        promotion = None
        if random.random() < 0.3:
            discount = round(total * random.choice([0.10, 0.15, 0.20, 0.25]), 2)
            promotion = f"{int(discount/total*100)}% OFF"
        
        # LOT_NUMBER, SERIAL_NUMBER (for applicable items)
        lot_number = None
        serial_number = None
        if category in ['beauty', 'health_wellness', 'food_beverage'] and random.random() < 0.4:
            lot_number = self.fake.bothify(text='LOT##??####')
        if category in ['electronics', 'jewelry'] and random.random() < 0.3:
            serial_number = self.fake.bothify(text='SN##########')
        
        # WEIGHT (for applicable items)
        weight = None
        if unit in ['lb', 'oz', 'kg']:
            weight = round(random.uniform(0.5, 5.0), 2)
        
        return RetailLineItem(
            description=description,
            quantity=quantity,
            unit_price=unit_price,
            total=total,
            upc=upc,
            sku=sku,
            unit=unit,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            discount=discount,
            lot_number=lot_number,
            serial_number=serial_number,
            weight=weight,
            promotion=promotion,
            rewards_earned=random.randint(1, 20) if random.random() < 0.4 else None
        )
    
    def generate_pos_receipt(self, 
                            store_type: str = 'fashion',
                            min_items: int = 3,
                            max_items: int = 8,
                            locale: Optional[str] = None) -> RetailReceiptData:
        """Generate a complete POS receipt with ALL 37 entities"""
        
        receipt = RetailReceiptData()
        
        # Randomly select locale if not provided
        if locale is None:
            locale_weights = {
                'en_US': 0.40, 'en_GB': 0.15, 'en_CA': 0.10, 'en_AU': 0.08,
                'fr_CA': 0.07, 'fr_FR': 0.05, 'es_ES': 0.05, 'es_MX': 0.04,
                'de_DE': 0.04, 'zh_CN': 0.02
            }
            locales = list(locale_weights.keys())
            weights = list(locale_weights.values())
            locale = random.choices(locales, weights=weights)[0]
        
        # Store locale in receipt data for rendering
        receipt.locale = locale
        
        # Document metadata
        receipt.doc_type = "Receipt"
        receipt.invoice_number = self.fake.bothify(text='REC-######')
        # Keep date in ISO format - will be formatted during rendering
        receipt.invoice_date = self.fake.date_this_year().strftime('%Y-%m-%d')
        # ORDER_DATE typically not on POS receipts
        
        # Merchant information
        receipt.supplier_name = random.choice(self.store_types.get(store_type, self.store_types['fashion']))
        receipt.supplier_address = self.fake.address().replace('\n', ', ')
        receipt.supplier_phone = self.fake.phone_number()
        receipt.supplier_email = self.fake.company_email()
        receipt.store_website = f"www.{receipt.supplier_name.lower().replace(' ', '')}.com"
        
        # Customer information (optional for POS - 40% have member info)
        if random.random() < 0.4:
            receipt.buyer_name = self.fake.name()
            receipt.buyer_phone = self.fake.phone_number()
            receipt.buyer_email = self.fake.email()
            receipt.account_number = self.fake.bothify(text='MEM######')  # ACCOUNT_NUMBER
            receipt.customer_id = receipt.account_number
        
        # Retail identifiers
        receipt.register_number = str(random.randint(1, 20))  # REGISTER_NUMBER
        receipt.cashier_id = self.fake.bothify(text='CSH###')  # CASHIER_ID
        receipt.transaction_number = self.fake.bothify(text='TRN#######')
        receipt.transaction_time = self.fake.time(pattern='%I:%M %p')
        
        # Generate line items
        num_items = random.randint(min_items, max_items)
        for _ in range(num_items):
            item = self.generate_line_item(category=store_type)
            receipt.line_items.append(item)
        
        # Calculate totals
        receipt.subtotal = sum(item.total for item in receipt.line_items)  # SUBTOTAL
        receipt.discount = sum(item.discount for item in receipt.line_items)  # DISCOUNT
        receipt.total_discount = receipt.discount
        
        # TAX_RATE, TAX_AMOUNT
        receipt.tax_rate = random.choice([6.5, 7.5, 8.0, 8.25, 9.0])
        taxable_amount = receipt.subtotal - receipt.discount
        receipt.tax_amount = round(taxable_amount * (receipt.tax_rate / 100), 2)
        
        # Tip (for QSR)
        if store_type == 'qsr' and random.random() < 0.6:
            receipt.tip_amount = round(receipt.subtotal * random.choice([0.15, 0.18, 0.20]), 2)
        
        # TOTAL_AMOUNT
        receipt.total_amount = round(receipt.subtotal - receipt.discount + receipt.tax_amount + receipt.tip_amount, 2)
        
        # CURRENCY
        receipt.currency = "$"
        
        # PAYMENT_METHOD
        receipt.payment_method = random.choice(self.payment_methods)
        
        # PAYMENT_TERMS (card details, approval codes)
        if receipt.payment_method in ["Visa", "Mastercard", "American Express", "Discover", "Debit Card"]:
            receipt.card_type = receipt.payment_method
            receipt.card_last_four = str(random.randint(1000, 9999))
            receipt.approval_code = self.fake.bothify(text='AUTH######')
            receipt.transaction_id = self.fake.bothify(text='TXN##########')
            receipt.payment_terms = f"Card ending in {receipt.card_last_four}, Auth: {receipt.approval_code}"
        elif receipt.payment_method == "Cash":
            receipt.cash_tendered = round(receipt.total_amount + random.uniform(0, 20), 2)
            receipt.change_amount = round(receipt.cash_tendered - receipt.total_amount, 2)
            receipt.payment_terms = f"Cash tendered: ${receipt.cash_tendered:.2f}"
        elif receipt.payment_method == "Gift Card":
            receipt.payment_terms = f"Gift card ending in {random.randint(1000, 9999)}"
        
        # TABLE (line items table)
        receipt.has_table = True
        
        # TERMS_AND_CONDITIONS
        receipt.terms_and_conditions = "All sales final. No refunds on sale items."
        
        # NOTE
        receipt.note = random.choice([
            "Thank you for shopping with us!",
            "Have a great day!",
            "We appreciate your business!",
            None
        ])
        
        # Loyalty
        if receipt.account_number:
            receipt.loyalty_points_earned = random.randint(10, 100)
            receipt.loyalty_points_balance = random.randint(100, 5000)
            receipt.loyalty_rewards_available = random.randint(0, 3)
        
        # Survey
        if random.random() < 0.5:
            receipt.survey_url = f"survey.{receipt.supplier_name.lower().replace(' ', '')}.com"
            receipt.survey_code = self.fake.bothify(text='SRV######')
        
        # Barcode
        receipt.barcode_value = receipt.invoice_number
        
        return receipt
    
    def generate_online_order(self, 
                             store_type: str = 'fashion',
                             min_items: int = 2,
                             max_items: int = 5,
                             locale: Optional[str] = None) -> RetailReceiptData:
        """Generate an online order/invoice with ALL 37 entities"""
        
        receipt = self.generate_pos_receipt(store_type=store_type, min_items=min_items, max_items=max_items, locale=locale)
        
        # Override for online orders
        receipt.doc_type = "Invoice"
        receipt.invoice_number = self.fake.bothify(text='ORD-######')
        
        # ORDER_DATE (for online orders) - keep in ISO format
        order_date = self.fake.date_between(start_date='-30d', end_date='today')
        receipt.order_date = order_date.strftime('%Y-%m-%d')
        receipt.invoice_date = (order_date + timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d')
        
        # Customer information (REQUIRED for online orders)
        receipt.buyer_name = self.fake.name()
        receipt.buyer_address = self.fake.address().replace('\n', ', ')
        receipt.buyer_phone = self.fake.phone_number()
        receipt.buyer_email = self.fake.email()
        receipt.account_number = self.fake.bothify(text='ACCT######')
        
        # TRACKING_NUMBER (for online orders)
        receipt.tracking_number = self.fake.bothify(text='1Z###??########')
        
        # Remove POS-specific fields
        receipt.register_number = None
        receipt.cashier_id = None
        receipt.transaction_time = None
        
        return receipt
    
    def to_dict(self, receipt: RetailReceiptData) -> Dict[str, Any]:
        """Convert RetailReceiptData to dictionary for template rendering"""
        return {
            # Locale for formatting
            'locale': receipt.locale,
            
            # Document metadata
            'doc_type': receipt.doc_type,
            'invoice_number': receipt.invoice_number,
            'invoice_date': receipt.invoice_date,
            'order_date': receipt.order_date,
            
            # Merchant
            'supplier_name': receipt.supplier_name,
            'supplier_address': receipt.supplier_address,
            'supplier_phone': receipt.supplier_phone,
            'supplier_email': receipt.supplier_email,
            'store_website': receipt.store_website,
            
            # Customer
            'buyer_name': receipt.buyer_name,
            'buyer_address': receipt.buyer_address,
            'buyer_phone': receipt.buyer_phone,
            'buyer_email': receipt.buyer_email,
            'customer_id': receipt.customer_id,
            'account_number': receipt.account_number,
            
            # Financial totals
            'currency': receipt.currency,
            'subtotal': f"${receipt.subtotal:.2f}",
            'tax_amount': f"${receipt.tax_amount:.2f}",
            'tax_rate': f"{receipt.tax_rate:.2f}",
            'total_amount': f"${receipt.total_amount:.2f}",
            'discount': f"${receipt.discount:.2f}",
            'total_discount': receipt.total_discount,
            'tip_amount': receipt.tip_amount,
            
            # Payment
            'payment_method': receipt.payment_method,
            'payment_terms': receipt.payment_terms,
            'card_type': receipt.card_type,
            'card_last_four': receipt.card_last_four,
            'approval_code': receipt.approval_code,
            'transaction_id': receipt.transaction_id,
            'cash_tendered': f"${receipt.cash_tendered:.2f}" if receipt.cash_tendered else None,
            'change_amount': f"${receipt.change_amount:.2f}" if receipt.change_amount else None,
            
            # Line items
            'line_items': [
                {
                    'description': item.description,
                    'quantity': item.quantity,
                    'unit_price': f"${item.unit_price:.2f}",
                    'total': f"${item.total:.2f}",
                    'upc': item.upc,
                    'sku': item.sku,
                    'unit': item.unit,
                    'tax_rate': item.tax_rate,
                    'tax_amount': item.tax_amount,
                    'discount': f"${item.discount:.2f}" if item.discount > 0 else None,
                    'lot_number': item.lot_number,
                    'serial_number': item.serial_number,
                    'weight': item.weight,
                    'promotion': item.promotion,
                    'rewards_earned': item.rewards_earned
                }
                for item in receipt.line_items
            ],
            'has_table': receipt.has_table,
            
            # Retail identifiers
            'register_number': receipt.register_number,
            'cashier_id': receipt.cashier_id,
            'cashier_name': f"Cashier {receipt.cashier_id}" if receipt.cashier_id else None,
            'tracking_number': receipt.tracking_number,
            'transaction_number': receipt.transaction_number,
            'transaction_time': receipt.transaction_time,
            
            # Miscellaneous
            'terms_and_conditions': receipt.terms_and_conditions,
            'note': receipt.note,
            'return_policy': receipt.return_policy,
            'footer_message': receipt.footer_message,
            
            # Loyalty
            'loyalty_points_earned': receipt.loyalty_points_earned,
            'loyalty_points_balance': receipt.loyalty_points_balance,
            'loyalty_rewards_available': receipt.loyalty_rewards_available,
            
            # Survey
            'survey_url': receipt.survey_url,
            'survey_code': receipt.survey_code,
            
            # Barcode
            'barcode_value': receipt.barcode_value,
            'barcode_image': receipt.barcode_image
        }


if __name__ == '__main__':
    # Test generation with Shopify-focused categories
    generator = RetailDataGenerator(seed=42)
    
    print("=== POS Receipt (Fashion Store) ===")
    pos_receipt = generator.generate_pos_receipt(store_type='fashion', min_items=3, max_items=6)
    print(f"Receipt: {pos_receipt.invoice_number}")
    print(f"Store: {pos_receipt.supplier_name}")
    print(f"Register: {pos_receipt.register_number}, Cashier: {pos_receipt.cashier_id}")
    print(f"Items: {len(pos_receipt.line_items)}")
    print(f"Subtotal: ${pos_receipt.subtotal:.2f}")
    print(f"Tax ({pos_receipt.tax_rate}%): ${pos_receipt.tax_amount:.2f}")
    print(f"Total: ${pos_receipt.total_amount:.2f}")
    print(f"Payment: {pos_receipt.payment_method} - {pos_receipt.payment_terms}")
    
    print("\n=== Online Order (Beauty Store) ===")
    online_order = generator.generate_online_order(store_type='beauty', min_items=2, max_items=4)
    print(f"Order: {online_order.invoice_number}")
    print(f"Customer: {online_order.buyer_name}")
    print(f"Tracking: {online_order.tracking_number}")
    print(f"Total: ${online_order.total_amount:.2f}")
