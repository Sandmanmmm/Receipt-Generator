"""
Retail-Specific Data Generator
Generates complete POS receipt and e-commerce order data with ALL 37 retail entities
Production-ready with realistic randomization patterns
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date
from decimal import Decimal
import random
from faker import Faker
from generators.production_randomizer import ProductionRandomizer
from generators.brand_name_generator import BrandNameGenerator
from generators.logo_generator import LogoGenerator
from generators.purchase_order import PurchaseOrder, PurchaseOrderLineItem, POType, POStatus


@dataclass
class RetailLineItem:
    """Retail line item with complete entity coverage + specialized fields"""
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
    
    # === FUEL STATION FIELDS ===
    is_fuel: bool = False
    fuel_grade: Optional[str] = None  # "Regular", "Premium", "Diesel"
    octane_rating: Optional[str] = None  # "87", "89", "93"
    gallons: Optional[float] = None
    price_per_gallon: Optional[float] = None
    rewards_discount: Optional[float] = None  # Discount per gallon
    total_savings: Optional[float] = None
    
    # === PHARMACY FIELDS ===
    rx_number: Optional[str] = None  # Prescription number
    drug_name: Optional[str] = None
    strength: Optional[str] = None  # "500mg", "10mg", etc.
    form: Optional[str] = None  # "Tablets", "Capsules", "Liquid"
    prescriber: Optional[str] = None  # Doctor name
    refills_remaining: Optional[int] = None
    refill_after_date: Optional[str] = None
    insurance_covered: bool = False
    insurance_paid: Optional[float] = None
    copay: Optional[float] = None
    insurance_savings: Optional[float] = None
    
    # === ELECTRONICS FIELDS ===
    brand: Optional[str] = None
    model_number: Optional[str] = None
    tech_specs: Optional[Dict[str, str]] = None
    warranty_period: Optional[str] = None  # "1 Year", "2 Years"
    warranty_expiry: Optional[str] = None
    extended_warranty: bool = False
    extended_warranty_period: Optional[str] = None
    configuration: Optional[str] = None  # "16GB RAM, 512GB SSD"
    accessories: Optional[List[str]] = None
    
    # === FASHION/APPAREL FIELDS ===
    size: Optional[str] = None  # "M", "L", "XL", "10"
    color: Optional[str] = None
    style: Optional[str] = None
    material: Optional[str] = None  # "100% Cotton"
    care_instructions: Optional[str] = None
    fit: Optional[str] = None  # "Regular", "Slim"
    original_price: Optional[float] = None
    savings: Optional[float] = None
    personalization: Optional[str] = None
    
    # === GROCERY FIELDS ===
    organic: bool = False
    locally_grown: bool = False
    expiration_date: Optional[str] = None
    weight_unit: Optional[str] = None  # "lb", "oz", "kg"
    unit_measure: Optional[str] = None  # "each", "bunch", "bag"
    price_per_unit: Optional[float] = None
    price_per_weight: Optional[float] = None
    on_sale: bool = False
    substituted: bool = False
    original_item: Optional[str] = None
    
    # === HOME IMPROVEMENT FIELDS ===
    dimensions: Optional[str] = None  # "24x36 inches"
    finish: Optional[str] = None  # "Matte", "Gloss"
    color_hex: Optional[str] = None  # "#FFFFFF"
    coverage: Optional[str] = None  # "Covers 400 sq ft"
    capacity: Optional[str] = None  # "5 Gallons"
    measure_unit: Optional[str] = None  # "sq ft", "linear ft"
    price_per_measure: Optional[float] = None
    installation_required: bool = False
    warranty: Optional[str] = None
    
    # === WHOLESALE/BULK FIELDS ===
    units_per_case: Optional[int] = None
    case_quantity: Optional[int] = None
    price_per_case: Optional[float] = None
    cases_per_pallet: Optional[int] = None
    pallet_quantity: Optional[int] = None
    price_per_pallet: Optional[float] = None
    moq: Optional[int] = None  # Minimum Order Quantity
    lead_time: Optional[str] = None  # "3-5 business days"
    manufacturer: Optional[str] = None
    country_of_origin: Optional[str] = None
    
    # === SHOPIFY INVENTORY UPLOAD FIELDS ===
    mpn: Optional[str] = None  # Manufacturer Part Number
    msrp: Optional[float] = None  # Manufacturer Suggested Retail Price
    map_price: Optional[float] = None  # Minimum Advertised Price
    case_pack: Optional[int] = None  # Units per case (alias for units_per_case)
    volume_discount_percentage: Optional[float] = None
    volume_discount_amount: Optional[float] = None
    tiered_pricing: Optional[bool] = None
    pricing_tiers: Optional[List[Dict[str, Any]]] = None
    price_tier: Optional[str] = None
    show_tiered_pricing_table: bool = False
    stock_status: Optional[str] = None
    available_quantity: Optional[int] = None
    line_total: Optional[float] = None
    total_units: Optional[int] = None
    promotional_discount: Optional[float] = None
    member_discount: Optional[float] = None
    deposit: Optional[float] = None
    tiered_discount: Optional[float] = None
    total_price: Optional[float] = None
    
    # === DIGITAL PRODUCTS FIELDS ===
    product_type: Optional[str] = None  # "Software", "Game", "Subscription"
    platform: Optional[str] = None  # "Windows", "Mac", "Multi-platform"
    version: Optional[str] = None  # "2024", "Pro", "v3.5"
    license_type: Optional[str] = None  # "Perpetual", "Subscription", "Single-user"
    license_seats: Optional[int] = None
    subscription_period: Optional[str] = None  # "1 Month", "1 Year"
    renewal_date: Optional[str] = None
    auto_renew: bool = False
    download_url: Optional[str] = None
    access_url: Optional[str] = None
    activation_code: Optional[str] = None
    license_key: Optional[str] = None
    file_size: Optional[str] = None  # "1.2 GB"
    system_requirements: Optional[str] = None
    expiry_date: Optional[str] = None
    
    # === QSR (QUICK SERVICE RESTAURANT) FIELDS ===
    modifiers: Optional[List[str]] = None  # ["No pickles", "Extra cheese"]
    special_instructions: Optional[str] = None
    is_combo: bool = False
    combo_savings: Optional[float] = None
    
    # === MARKETPLACE FIELDS ===
    seller_name: Optional[str] = None
    seller_sku: Optional[str] = None
    condition: Optional[str] = None  # "New", "Used - Like New"
    variant: Optional[str] = None  # "Color: Blue, Size: M"
    
    # === SHOPIFY PRODUCT CLASSIFICATION FIELDS ===
    category: Optional[str] = None  # Product category for Shopify (e.g., "Apparel", "Electronics")
    collection: Optional[str] = None  # ITEM_COLLECTION - Product line/collection (e.g., "Spring 2024", "Pro Series")
    
    # === GENERAL ADDITIONAL FIELDS ===
    discount_description: Optional[str] = None
    name: Optional[str] = None  # Alias for description
    attributes: Optional[Dict[str, Any]] = None  # Generic key-value attributes


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
    supplier_logo: Optional[str] = None  # Logo path or data URI
    
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
    payment_method: Optional[str] = None  # PAYMENT_METHOD - set by context-aware generator
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
    
    # Marketplace seller info (order-level)
    seller_name: Optional[str] = None
    seller_username: Optional[str] = None
    seller_rating: Optional[float] = None
    seller_reviews: Optional[int] = None
    marketplace_name: Optional[str] = None
    
    # Wholesale B2B fields
    po_number: Optional[str] = None  # Purchase Order number
    shipping_terms: Optional[str] = None  # FOB, CIF, etc.
    shipping_cost: float = 0.0
    account_manager: Optional[str] = None  # Sales rep name
    bank_details: Optional[str] = None  # Wire transfer instructions
    tax_exempt: Optional[str] = None  # Tax exemption certificate number
    
    # E-commerce fields
    shipping_method: Optional[str] = None  # Standard, Express, etc.
    delivery_estimate: Optional[str] = None  # Estimated delivery date
    gift_message: Optional[str] = None  # Gift message
    notes: Optional[str] = None  # Order notes
    
    # Locale for formatting
    locale: str = 'en_US'


class RetailDataGenerator:
    """Generates retail-specific synthetic data ensuring ALL 37 entities appear"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        self.fake = Faker(locale)
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # Initialize logo generator
        self.logo_generator = LogoGenerator()
        
        # Retail product categories (optimized for Shopify use cases)
        # Expanded to 50+ items per category for production readiness
        self.fashion_items = [
            # Jeans & Denim (Branded)
            "Levi's 501 Original Jeans", "Levi's 511 Slim Fit Jeans", "Levi's 721 High Rise Skinny",
            "Wrangler Cowboy Cut Jeans", "Lee Relaxed Fit Jeans", "Calvin Klein Skinny Jeans",
            "American Eagle Jegging", "Gap 1969 Straight Jeans", "True Religion Skinny Jeans",
            "AG Farrah Skinny Ankle Jeans", "Madewell High-Rise Skinny", "7 For All Mankind Bootcut",
            # T-Shirts & Tops (Branded)
            "Hanes ComfortSoft Tee", "Gildan Heavy Cotton T-Shirt", "Champion Classic Tee",
            "Nike Dri-FIT Training Tee", "Adidas Essentials 3-Stripes Tee", "Under Armour Tech Tee",
            "Carhartt Workwear Pocket Tee", "Dickies Short Sleeve Work Shirt",
            "Uniqlo Supima Cotton Crew Neck", "H&M Organic Cotton V-Neck", "Zara Basic Crop Top",
            "Free People We The Free Tee", "Madewell Whisper Cotton V-Neck",
            # Hoodies & Sweatshirts (Branded)
            "Nike Sportswear Club Fleece Hoodie", "Champion Powerblend Pullover", "Adidas Essentials Hoodie",
            "Carhartt Rain Defender Hoodie", "The North Face Drew Peak Hoodie", "Patagonia Better Sweater",
            "Hanes EcoSmart Hoodie", "Russell Athletic Dri-Power Hoodie",
            # Athletic/Activewear (Branded)
            "Nike Pro Leggings", "Adidas Tiro Track Pants", "Lululemon Align High-Rise Pant",
            "Under Armour HeatGear Compression", "Athleta Salutation Stash Pocket", "Gymshark Vital Seamless",
            "Puma Training Tee", "Reebok CrossFit Speed Shorts", "Nike Dri-FIT Sports Bra",
            "Adidas Believe This 7/8 Tight", "Lululemon Wunder Train", "Outdoor Voices Exercise Dress",
            # Sneakers & Shoes (Specific Models)
            "Nike Air Force 1 '07", "Nike Air Max 90", "Nike Dunk Low", "Nike Blazer Mid '77",
            "Adidas Superstar", "Adidas Stan Smith", "Adidas Ultraboost 23", "Adidas Samba OG",
            "Converse Chuck Taylor All Star", "Vans Old Skool", "New Balance 574", "New Balance 990v5",
            "Puma Suede Classic", "Reebok Classic Leather", "On Cloud 5", "Hoka Clifton 9",
            "Allbirds Tree Runners", "Crocs Classic Clog", "Birkenstock Arizona Sandal",
            # Dresses (Branded/Style-Specific)
            "Zara Midi Wrap Dress", "H&M Puff Sleeve Dress", "Forever 21 Mini Bodycon Dress",
            "Free People Adella Maxi Slip", "Reformation Carina Dress", "Anthropologie Maeve Dress",
            "Gap Fit & Flare Dress", "Old Navy Tiered Maxi Dress", "Madewell Lightspun Dress",
            # Outerwear (Branded)
            "The North Face 1996 Retro Nuptse Jacket", "Patagonia Better Sweater Fleece",
            "Canada Goose Expedition Parka", "Arc'teryx Beta LT Jacket", "Columbia Heavenly Hooded Jacket",
            "Carhartt Detroit Jacket", "Levi's Sherpa Trucker Jacket", "Dickies Eisenhower Jacket",
            "Nike Sportswear Windrunner", "Adidas Adicolor SST Track Jacket",
            # Pants & Bottoms (Branded)
            "Dickies 874 Work Pants", "Carhartt Double Knee Pant", "Dockers Alpha Khaki",
            "Gap Khakis in Slim Fit", "Uniqlo Ultra Stretch Jeans", "J.Crew 770 Straight Chino",
            "Lululemon ABC Pant", "Bonobos Stretch Washed Chinos",
            # Shirts & Blouses (Branded)
            "Ralph Lauren Polo Shirt", "Tommy Hilfiger Classic Fit Oxford", "Brooks Brothers Non-Iron Shirt",
            "Lacoste L.12.12 Polo", "Uniqlo Linen Blend Shirt", "Everlane Clean Silk Blouse",
            # Basics & Underwear (Branded)
            "Hanes ComfortFlex Boxer Briefs", "Calvin Klein Cotton Stretch Boxer", "Fruit of the Loom Tank Top",
            "Champion Sports Bra", "Jockey Elance Bikini", "Tommy John Second Skin Trunk",
            # Sweaters & Cardigans (NEW)
            "J.Crew Cashmere Crewneck Sweater", "Everlane Cashmere Crew", "Uniqlo Extra Fine Merino Crew Neck",
            "Gap Cable-Knit Cardigan", "H&M Fine-Knit Turtleneck", "Banana Republic Italian Merino V-Neck",
            "L.L.Bean Classic Ragg Wool Sweater", "Eddie Bauer Lodge Pullover", "Lands' End Drifter Cardigan",
            # Skirts (NEW)
            "Zara Satin Midi Skirt", "H&M Pleated Mini Skirt", "Forever 21 Denim Skirt",
            "Madewell Tiered Maxi Skirt", "Anthropologie Maeve Bias Skirt", "Gap Fit and Flare Skirt",
            # Swimwear (NEW)
            "Speedo Endurance+ Swimsuit", "TYR Durafast Elite Suit", "Nike Essential Swim Shorts",
            "Billabong Boardshorts", "Roxy One Piece Swimsuit", "Victoria's Secret Bikini Set",
            "Lands' End Chlorine Resistant Suit", "Athleta Shirrendipity Bikini",
            # Sleepwear & Loungewear (NEW)
            "Victoria's Secret Pajama Set", "Gap Body Sleep Shorts", "Old Navy Cozy Lounge Pants",
            "Soma Cool Nights Pajamas", "Eberjey Gisele PJ Set", "UGG Robe", "Barefoot Dreams Cardigan",
            # Workwear & Uniforms (NEW)
            "Dickies Industrial Work Shirt", "Red Kap Performance Work Pants", "Carhartt Flame-Resistant Jeans",
            "Cherokee Workwear Scrubs", "Landau Medical Scrubs", "Dickies EDS Essentials Scrubs",
            "5.11 Tactical Pants", "Propper BDU Pants", "Wrangler Riggs Workwear Jeans"
        ]
        
        self.accessories_items = [
            # Bags
            "Leather Handbag", "Canvas Tote Bag", "Crossbody Bag", "Backpack",
            "Messenger Bag", "Clutch Purse", "Weekender Bag", "Laptop Bag",
            "Beach Bag", "Diaper Bag", "Fanny Pack", "Drawstring Bag",
            "Mini Backpack", "Sling Bag", "Bucket Bag", "Saddle Bag", "Hobo Bag",
            # Hats & Headwear
            "Baseball Cap", "Beanie", "Fedora", "Wide Brim Hat", "Bucket Hat",
            "Sun Hat", "Beret", "Trucker Hat",
            "Snapback Cap", "Dad Hat", "Visor", "Newsboy Cap", "Panama Hat",
            # Scarves & Wraps
            "Silk Scarf", "Infinity Scarf", "Blanket Scarf", "Pashmina Shawl",
            "Bandana Scarf", "Neck Gaiter", "Knit Scarf", "Cashmere Scarf",
            # Belts & Wallets
            "Leather Belt", "Canvas Belt", "Chain Belt", "Leather Wallet",
            "Card Holder", "Money Clip", "Coin Purse",
            "Woven Belt", "Reversible Belt", "Studded Belt", "Bifold Wallet", "Zip Wallet",
            # Eyewear
            "Sunglasses", "Reading Glasses", "Blue Light Glasses", "Sports Sunglasses",
            "Cat Eye Sunglasses", "Aviator Sunglasses", "Round Sunglasses", "Polarized Sunglasses",
            # Hair Accessories
            "Hair Clips Set", "Headband", "Scrunchie Set", "Hair Ties Pack",
            "Bobby Pins", "Barrettes",
            "Claw Clips", "Silk Scrunchies", "Padded Headband", "Pearl Headband",
            # Watches
            "Stainless Steel Watch", "Leather Watch", "Smart Watch", "Sport Watch",
            "Digital Watch", "Chronograph Watch", "Minimalist Watch", "Wooden Watch",
            # Other
            "Umbrella", "Keychain", "Phone Charm", "Luggage Tag", "Bandana",
            "Travel Organizer", "Pill Case", "Compact Mirror", "Gloves"
        ]
        
        self.jewelry_items = [
            # Necklaces
            "Sterling Silver Necklace", "Gold Chain Necklace", "Rose Gold Pendant",
            "Layered Necklace", "Choker Necklace", "Pearl Necklace", "Locket Necklace",
            "Bar Necklace", "Name Necklace", "Cross Necklace",
            "Initial Necklace", "Coin Necklace", "Heart Pendant", "Infinity Necklace", "Y-Necklace",
            # Earrings
            "Gold Hoop Earrings", "Gemstone Stud Earrings", "Drop Earrings",
            "Dangle Earrings", "Huggie Hoops", "Pearl Studs", "Crystal Earrings",
            "Threader Earrings", "Ear Cuffs",
            "Diamond Studs", "Tassel Earrings", "Chandelier Earrings", "Climber Earrings",
            # Bracelets
            "Pearl Bracelet", "Charm Bracelet", "Cuff Bracelet", "Bangle Set",
            "Tennis Bracelet", "Chain Bracelet", "Leather Bracelet", "Beaded Bracelet",
            "Friendship Bracelet", "Anklet",
            "Link Bracelet", "Wrap Bracelet", "Slide Bracelet", "Personalized Bracelet",
            # Rings
            "Diamond Ring", "Engagement Ring", "Wedding Band", "Stackable Rings",
            "Statement Ring", "Midi Ring", "Signet Ring", "Birthstone Ring",
            "Promise Ring", "Eternity Band",
            "Cocktail Ring", "Solitaire Ring", "Halo Ring", "Three-Stone Ring", "Band Ring Set",
            # Sets
            "Jewelry Set", "Bridal Set", "Necklace & Earring Set",
            "Bracelet & Earring Set", "Matching Ring Set", "Gift Box Set",
            # Body Jewelry
            "Belly Ring", "Nose Ring", "Toe Ring", "Body Chain",
            "Septum Ring", "Cartilage Earring", "Tragus Stud",
            # Other
            "Brooch Pin", "Lapel Pin", "Cufflinks", "Tie Clip",
            "Charm", "Chain Extender", "Jewelry Box", "Ring Holder"
        ]
        
        self.beauty_items = [
            # Skincare - Face
            "Organic Face Serum", "Hydrating Moisturizer", "Vitamin C Serum",
            "Retinol Cream", "Hyaluronic Acid Serum", "Facial Cleanser", "Toner",
            "Face Mask Set", "Sheet Mask", "Eye Cream", "Night Cream", "Day Cream",
            "Sunscreen SPF 50", "Exfoliating Scrub", "Micellar Water", "Essence",
            "Niacinamide Serum", "AHA BHA Toner", "Vitamin E Oil", "Rose Water Spray",
            "Collagen Mask", "Clay Mask", "Peel-Off Mask", "Sleeping Mask",
            # Skincare - Body
            "Body Butter", "Body Scrub", "Body Lotion", "Hand Cream",
            "Foot Cream", "Body Oil", "Bath Salts", "Body Wash",
            # Makeup - Face
            "BB Cream", "CC Cream", "Foundation", "Concealer", "Powder",
            "Blush", "Bronzer", "Highlighter", "Setting Spray", "Primer",
            "Contour Palette", "Color Corrector", "Setting Powder", "Face Primer",
            # Makeup - Eyes
            "Eye Shadow Palette", "Mascara", "Eyeliner", "Eyebrow Pencil",
            "Brow Gel", "Eye Primer", "False Lashes",
            "Liquid Eyeliner", "Gel Eyeliner", "Eyeshadow Stick", "Brow Pomade",
            # Makeup - Lips
            "Matte Lipstick", "Lip Gloss", "Lip Liner", "Liquid Lipstick",
            "Lip Balm", "Lip Stain", "Lip Scrub",
            "Lip Plumper", "Tinted Lip Balm", "Lip Mask", "Lip Oil",
            # Tools & Brushes
            "Makeup Brush Set", "Beauty Blender", "Eyelash Curler", "Tweezers",
            "Makeup Sponge", "Brush Cleaner",
            "Facial Roller", "Gua Sha Tool", "LED Face Mask", "Makeup Mirror",
            # Nails
            "Nail Polish", "Gel Polish", "Nail File Set", "Cuticle Oil",
            "Press-On Nails", "Nail Art Kit", "UV Lamp", "Nail Strengthener",
            # Haircare - Shampoo & Conditioner
            "Argan Oil Shampoo", "Purple Shampoo", "Clarifying Shampoo",
            "Keratin Conditioner", "Deep Conditioning Mask", "Hair Mask",
            "Scalp Treatment", "Hair Detangler Spray",
            # Haircare - Styling
            "Hair Styling Cream", "Hair Oil", "Dry Shampoo", "Leave-In Conditioner",
            "Hair Mousse", "Heat Protectant Spray", "Hair Gel", "Texture Spray",
            "Hair Serum", "Anti-Frizz Cream", "Curl Defining Cream",
            # Haircare - Tools
            "Hair Dryer Brush", "Curling Iron", "Flat Iron", "Hair Diffuser",
            # Fragrance
            "Perfume 50ml", "Body Spray", "Cologne", "Roll-On Perfume",
            "Fragrance Gift Set", "Travel Size Perfume", "Body Mist"
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
            "Dreamcatcher", "Macrame Wall Hanging", "Woven Basket", "Decorative Tray",
            # Textiles
            "Bath Towel Set", "Hand Towel", "Washcloth Set", "Bath Mat",
            "Shower Curtain", "Table Runner", "Placemats Set", "Napkin Set",
            "Area Rug", "Door Mat", "Kitchen Towel Set",
            "Duvet Cover Set", "Fitted Sheet Set", "Pillowcase Set", "Quilt",
            # Storage & Organization
            "Storage Basket", "Storage Bin", "Shelf Organizer", "Drawer Divider",
            "Jewelry Box", "Makeup Organizer", "Closet Organizer",
            "Shoe Rack", "Over-Door Organizer", "Under-Bed Storage", "Cable Organizer",
            # Kitchen & Dining
            "Ceramic Dinnerware Set", "Glass Drinkware Set", "Cutlery Set",
            "Serving Platter", "Mixing Bowl Set", "Measuring Cup Set",
            "Coffee Mug Set", "Wine Glass Set", "Champagne Flutes",
            "Salt & Pepper Shakers", "Kitchen Utensil Set", "Cutting Board Set",
            # Garden & Outdoor
            "Plant Pot", "Planter Box", "Watering Can", "Garden Tools Set",
            "Plant Stand", "Hanging Planter", "Seed Starter Kit", "Garden Gloves",
            "Garden Hose", "Plant Markers", "Pruning Shears", "Garden Kneeler",
            "Bird Feeder", "Wind Chimes", "Solar Garden Lights", "Garden Statue",
            # Furniture & Large Decor
            "Accent Chair Cushion", "Bar Stool Cushion", "Bench Cushion",
            "Coat Rack", "Wall Shelf", "Floating Shelf Set", "Bookshelf"
        ]
        
        self.sports_fitness_items = [
            # Equipment
            "Yoga Mat", "Exercise Mat", "Foam Roller", "Resistance Bands",
            "Resistance Loop Bands", "Jump Rope", "Exercise Ball", "Medicine Ball",
            "Dumbbells 10lb", "Dumbbells 20lb", "Kettlebell", "Weight Set",
            "Pull Up Bar", "Push Up Bars", "Ab Wheel", "Balance Ball",
            "Yoga Blocks", "Yoga Strap", "Massage Ball",
            "Adjustable Dumbbell Set", "Barbell Set", "Weight Bench", "Squat Rack",
            "Battle Ropes", "Plyo Box", "Slam Ball", "TRX Suspension Trainer",
            # Cardio Equipment
            "Treadmill Compact", "Exercise Bike", "Rowing Machine", "Elliptical Trainer",
            "Mini Stepper", "Speed Ladder", "Agility Cones", "Hurdle Set",
            # Apparel
            "Gym Bag", "Duffle Bag", "Workout Gloves", "Lifting Straps",
            "Compression Socks", "Compression Sleeves", "Sweatband Set",
            "Athletic Headband", "Workout Towel",
            "Weight Lifting Belt", "Knee Sleeves", "Wrist Wraps", "Ankle Weights",
            # Nutrition
            "Protein Powder", "Protein Bar Box", "Pre-Workout", "BCAA Powder",
            "Creatine", "Protein Shaker", "Pill Organizer", "Supplement Container",
            # Hydration
            "Water Bottle", "Insulated Water Bottle", "Sport Water Bottle",
            "Hydration Pack", "Electrolyte Powder",
            "Gallon Water Jug", "Insulated Tumbler 30oz", "Collapsible Water Bottle",
            # Tech & Accessories
            "Fitness Tracker", "Heart Rate Monitor", "Stopwatch", "Pedometer",
            "Armband Phone Holder", "Workout Timer", "Bluetooth Earbuds",
            "Running Belt", "Sport Headphones", "Activity Tracker Ring",
            # Recovery
            "Ice Pack", "Heating Pad", "Massage Gun", "Kinesiology Tape",
            "Foam Roller Vibrating", "Percussion Massager", "Muscle Roller Stick",
            "Compression Boots", "Ice Bath Tub", "Sauna Blanket",
            # Outdoor & Sports
            "Basketball", "Soccer Ball", "Volleyball", "Football",
            "Tennis Racket", "Badminton Set", "Pickleball Paddle Set",
            "Golf Balls", "Frisbee", "Swimming Goggles", "Swim Cap"
        ]
        
        self.pet_supplies_items = [
            # Food & Treats
            "Dog Food", "Cat Food", "Dog Treats", "Cat Treats", "Dental Chews",
            "Bully Sticks", "Rawhide Bones", "Freeze-Dried Treats",
            "Puppy Food", "Kitten Food", "Senior Dog Food", "Grain-Free Cat Food",
            "Wet Dog Food", "Wet Cat Food", "Jerky Treats", "Training Treats",
            # Bowls & Feeders
            "Food Bowl Set", "Water Bowl", "Elevated Feeder", "Automatic Feeder",
            "Slow Feeder Bowl", "Travel Bowl",
            "Stainless Steel Bowl", "Ceramic Pet Bowl", "Silicone Mat", "Fountain Water Bowl",
            # Toys
            "Cat Toy", "Dog Toy", "Chew Toys", "Rope Toy", "Plush Toy",
            "Interactive Toy", "Puzzle Toy", "Ball Launcher", "Catnip Toys",
            "Feather Wand", "Laser Pointer",
            "Kong Toy", "Squeaky Toy", "Tug Toy", "Fetch Ball", "Cat Tunnel",
            # Beds & Furniture
            "Pet Bed", "Dog Bed", "Cat Bed", "Pet Blanket", "Pet Mat",
            "Cat Tree", "Scratching Post", "Pet Stairs", "Pet Ramp",
            "Orthopedic Dog Bed", "Heated Pet Bed", "Window Perch", "Cat Condo",
            # Leashes & Collars
            "Dog Collar", "Cat Collar", "Leash", "Retractable Leash", "Harness",
            "ID Tag", "Collar Charm",
            "Martingale Collar", "No-Pull Harness", "Reflective Leash", "Hands-Free Leash",
            # Grooming
            "Pet Shampoo", "Grooming Brush", "Nail Clippers", "Pet Wipes",
            "Deshedding Tool", "Toothbrush Set", "Ear Cleaner",
            "Slicker Brush", "Undercoat Rake", "Nail Grinder", "Grooming Glove",
            # Litter & Cleanup
            "Cat Litter", "Litter Box", "Litter Scoop", "Waste Bags", "Pet Odor Spray",
            "Clumping Litter", "Hooded Litter Box", "Self-Cleaning Litter Box", "Poop Scooper",
            # Travel & Carriers
            "Pet Carrier", "Travel Crate", "Car Seat Cover", "Pet Seatbelt",
            "Airline-Approved Carrier", "Backpack Carrier", "Pet Stroller", "Travel Bag",
            # Health & Wellness
            "Flea & Tick Collar", "Vitamin Supplement", "Joint Support Chews", "Calming Treats"
        ]
        
        self.books_media_items = [
            # Books
            "Hardcover Book", "Paperback Book", "Comic Book", "Graphic Novel",
            "Coloring Book", "Activity Book", "Cookbook", "Self-Help Book",
            "Children's Book", "Young Adult Novel", "Fantasy Novel", "Mystery Book",
            "Biography", "Memoir", "History Book", "Science Fiction Novel",
            "Romance Novel", "Thriller", "Horror Novel", "Poetry Book",
            # Journals & Notebooks
            "Journal", "Bullet Journal", "Travel Journal", "Gratitude Journal",
            "Notebook", "Sketchbook", "Composition Notebook", "Spiral Notebook",
            "Planner", "Daily Planner", "Weekly Planner", "Agenda",
            "Leather Journal", "Dotted Notebook", "Grid Notebook", "Academic Planner",
            # Art & Prints
            "Art Print", "Poster", "Canvas Print", "Framed Print",
            "Photography Print", "Vintage Poster",
            "Movie Poster", "Music Poster", "Motivational Print", "Quote Print",
            # Stationery
            "Pen Set", "Pencil Set", "Marker Set", "Highlighter Set",
            "Sticky Notes", "Washi Tape", "Sticker Pack", "Bookmark Set",
            "Greeting Cards", "Thank You Cards", "Note Cards",
            "Fountain Pen", "Gel Pen Set", "Mechanical Pencil", "Calligraphy Set",
            # Accessories
            "Book Light", "Reading Light", "Book Stand", "Bookends",
            "Book Sleeve", "Page Markers",
            "Reading Pillow", "Book Weight", "Bookmark Light", "Book Holder",
            # Subscriptions & Media
            "Magazine Subscription", "Digital Download", "E-Book", "Audiobook",
            "Gift Card", "Book Box Subscription"
        ]
        
        self.toys_games_items = [
            # Puzzles
            "Puzzle 1000pc", "Puzzle 500pc", "3D Puzzle", "Jigsaw Puzzle",
            "Brain Teaser", "Puzzle Mat",
            "Puzzle 2000pc", "Wooden Puzzle", "Metal Puzzle", "Escape Room Puzzle",
            # Board Games & Cards
            "Board Game", "Strategy Game", "Party Game", "Family Game",
            "Card Game", "Playing Cards", "Trivia Game", "Chess Set",
            "Checkers", "Backgammon", "Dice Set",
            "Monopoly", "Scrabble", "Clue", "Risk", "Catan", "Ticket to Ride",
            "Uno Card Game", "Phase 10", "Exploding Kittens", "Cards Against Humanity",
            # Action Figures & Dolls
            "Action Figure", "Doll", "Fashion Doll", "Baby Doll",
            "Doll House", "Action Figure Set", "Collectible Figure",
            "Barbie Doll", "Hot Wheels Car", "Matchbox Car Set", "Funko Pop",
            # Building & Construction
            "Building Blocks", "LEGO Set", "Construction Set", "Marble Run",
            "Magnetic Tiles", "Wooden Blocks",
            "LEGO Technic Set", "LEGO Star Wars Set", "K'NEX Building Set", "Mega Bloks",
            # Stuffed Animals
            "Stuffed Animal", "Plush Toy", "Teddy Bear", "Unicorn Plush",
            "Squishmallow", "Build-A-Bear", "Giant Teddy Bear", "Weighted Plush",
            # Remote Control & Tech
            "Remote Control Car", "RC Truck", "Drone", "Robot Toy",
            "RC Helicopter", "RC Boat", "Robot Dog", "Coding Robot",
            # Arts & Crafts
            "Art Supplies", "Craft Kit", "Paint Set", "Crayon Set",
            "Play-Doh Set", "Modeling Clay", "Bead Kit", "Sewing Kit",
            "Origami Paper", "Coloring Set", "Watercolor Set",
            "Slime Kit", "Friendship Bracelet Kit", "Diamond Painting Kit", "Spin Art Kit",
            # Outdoor & Active
            "Toy Car", "Toy Train", "Toy Kitchen", "Play Tent",
            "Water Toys", "Sand Toys", "Bubble Machine", "Kite",
            "Nerf Blaster", "Water Gun", "Sidewalk Chalk", "Jump Rope",
            # Educational
            "STEM Kit", "Science Kit", "Learning Toy", "Flash Cards",
            "Math Flash Cards", "Alphabet Puzzle", "Microscope Kit", "Chemistry Set"
        ]
        
        self.food_beverage_items = [
            # Coffee (Branded)
            "Starbucks Pike Place Ground Coffee 12oz", "Dunkin' Original Blend", "Folgers Classic Roast",
            "Lavazza Super Crema Espresso", "Peet's Coffee Major Dickason's Blend",
            "Death Wish Coffee Strong 16oz", "Kicking Horse 454 Horse Power", "Café Bustelo Espresso",
            "Nespresso Vertuo Capsules", "Starbucks Veranda Blend K-Cups",
            # Tea (Branded)
            "Twinings English Breakfast Tea", "Bigelow Green Tea", "Tazo Zen Green Tea",
            "Celestial Seasonings Sleepytime", "Lipton Black Tea", "Harney & Sons Hot Cinnamon",
            "Republic of Tea Honey Ginseng", "Yogi Tea Bedtime",
            # Beverages (Branded)
            "Red Bull Energy Drink 12-Pack", "Monster Energy Ultra", "Gatorade Zero 12-Pack",
            "Coca-Cola 12-Pack Cans", "Pepsi Zero Sugar 12-Pack", "Dr Pepper 12-Pack",
            "Sparkling Ice Black Raspberry", "LaCroix Sparkling Water", "Perrier Carbonated Water",
            "Vitaminwater Zero Sugar", "Bodyarmor Sports Drink", "Liquid Death Mountain Water",
            # Snacks - Chips (Branded)
            "Lay's Classic Potato Chips", "Doritos Nacho Cheese", "Cheetos Crunchy",
            "Pringles Original", "Ruffles Cheddar & Sour Cream", "SunChips Harvest Cheddar",
            "Tostitos Scoops Tortilla Chips", "Fritos Original Corn Chips", "Kettle Brand Sea Salt",
            "Cape Cod Original Chips", "Popchips BBQ", "Veggie Straws Sea Salt",
            # Snacks - Bars & Granola (Branded)
            "Clif Bar Variety Pack 12-Count", "Kind Bars Dark Chocolate", "RXBAR Protein Bar",
            "Nature Valley Crunchy Granola", "Quest Protein Bar Variety", "Lärabar Cashew Cookie",
            "GoMacro MacroBar", "Perfect Bar Peanut Butter", "Built Bar Protein",
            "Nature's Bakery Fig Bar", "Kellogg's Nutri-Grain", "Quaker Chewy Granola Bars",
            # Snacks - Crackers & Pretzels (Branded)
            "Ritz Original Crackers", "Wheat Thins Original", "Triscuit Original",
            "Goldfish Cheddar Crackers", "Cheez-It Original", "Town House Crackers",
            "Snyder's of Hanover Pretzels", "Rold Gold Tiny Twists", "Pepperidge Farm Goldfish",
            # Nuts & Seeds (Branded)
            "Planters Dry Roasted Peanuts", "Blue Diamond Almonds", "Wonderful Pistachios",
            "Fisher Deluxe Mixed Nuts", "Sahale Snacks Glazed Mix", "Emerald 100 Calorie Packs",
            # Candy & Chocolate (Branded)
            "M&M's Milk Chocolate", "Snickers Bar", "Reese's Peanut Butter Cups",
            "Kit Kat Bar", "Hershey's Milk Chocolate", "Twix Caramel Cookie",
            "Lindt Excellence Dark Chocolate", "Ghirardelli Chocolate Squares", "Godiva Chocolate Truffles",
            "Ferrero Rocher Collection", "Toblerone Swiss Chocolate", "Milka Alpine Milk",
            # Condiments & Sauces (Branded)
            "Heinz Tomato Ketchup 32oz", "French's Yellow Mustard", "Hellmann's Real Mayonnaise",
            "Kraft Original BBQ Sauce", "Sweet Baby Ray's BBQ", "Hidden Valley Ranch Dressing",
            "Tabasco Original Red Sauce", "Sriracha Hot Chili Sauce", "Frank's RedHot Original",
            "Classico Tomato Basil Pasta Sauce", "Rao's Marinara Sauce", "Prego Traditional",
            # Spreads & Nut Butters (Branded)
            "Jif Creamy Peanut Butter", "Skippy Chunky Peanut Butter", "Justin's Almond Butter",
            "Nutella Hazelnut Spread", "Smucker's Strawberry Jam", "Welch's Grape Jelly",
            "Bonne Maman Raspberry Preserves", "Teddie All Natural Peanut Butter",
            # Cereals & Breakfast (Branded)
            "Cheerios Original", "Lucky Charms", "Frosted Flakes", "Cinnamon Toast Crunch",
            "Honey Nut Cheerios", "Froot Loops", "Special K Original", "Raisin Bran",
            "Kashi GO Original", "Nature's Path Organic Flax", "Bob's Red Mill Oatmeal",
            # Baking & Cooking (Branded)
            "King Arthur All-Purpose Flour", "Bob's Red Mill Almond Flour", "C&H Pure Cane Sugar",
            "Domino Light Brown Sugar", "McCormick Pure Vanilla Extract", "Ghirardelli Cocoa Powder",
            # Beverages - Hot Chocolate & Instant (Branded)
            "Swiss Miss Hot Cocoa Mix", "Nesquik Chocolate Powder", "Ovaltine Chocolate Malt",
            # Pasta & Grains (Branded)
            "Barilla Penne Pasta", "De Cecco Spaghetti", "Banza Chickpea Pasta",
            "Lundberg Organic Brown Rice", "Uncle Ben's Ready Rice", "Near East Rice Pilaf",
            "Quinoa Ancient Harvest", "Bob's Red Mill Steel Cut Oats",
            # Canned & Packaged (Branded)
            "Campbell's Tomato Soup", "Progresso Chicken Noodle", "Amy's Organic Lentil Soup",
            "Bush's Baked Beans", "Goya Black Beans", "S&W Chickpeas",
            "Bumble Bee Tuna", "StarKist Albacore Tuna", "Wild Planet Salmon",
            # Frozen Foods (Branded)
            "Stouffer's Lasagna", "Lean Cuisine Chicken", "Birds Eye Vegetables",
            "Ben & Jerry's Ice Cream", "Häagen-Dazs Vanilla", "Talenti Gelato",
            "Ore-Ida French Fries", "Eggo Waffles", "Jimmy Dean Breakfast Sandwich",
            # Dairy Alternatives (Branded)
            "Oatly Oat Milk", "Silk Almond Milk", "So Delicious Coconut Milk",
            "Chobani Oat Milk", "Califia Farms Almond Creamer", "Ripple Pea Protein Milk",
            # Health Foods (Branded)
            "Orgain Organic Protein Shake", "Vega Protein Powder", "Ancient Nutrition Collagen",
            "Bulletproof Brain Octane Oil", "Primal Kitchen Mayo", "Sir Kensington's Ketchup",
            # International Foods (Branded)
            "Kikkoman Soy Sauce", "Lee Kum Kee Oyster Sauce", "Thai Kitchen Coconut Milk",
            "Goya Adobo Seasoning", "Old El Paso Taco Kit", "Annie Chun's Rice Noodles"
        ]
        
        # Grocery delivery items organized by temperature zone
        self.grocery_frozen_items = [
            "Frozen Pizza Pepperoni", "Frozen Chicken Breasts", "Frozen Ground Beef",
            "Frozen Fish Fillets", "Frozen Vegetables Mixed", "Frozen Broccoli Florets",
            "Ben & Jerry's Ice Cream Pint", "Häagen-Dazs Ice Cream", "Ice Cream Sandwiches",
            "Frozen Waffles", "Frozen French Fries", "Frozen Chicken Nuggets",
            "Frozen Burrito", "Frozen Lasagna", "Frozen Meatballs"
        ]
        
        self.grocery_refrigerated_items = [
            "Milk Whole Gallon", "Milk 2% Half Gallon", "Almond Milk", "Oat Milk",
            "Greek Yogurt Plain", "Yogurt Strawberry", "Cottage Cheese", "Sour Cream",
            "Cheddar Cheese Block", "Mozzarella Shredded", "Cream Cheese",
            "Butter Salted", "Eggs Large Dozen", "Eggs Extra Large",
            "Fresh Chicken Breast", "Ground Beef 80/20", "Pork Chops", "Bacon",
            "Deli Ham Sliced", "Turkey Breast Deli", "Refrigerated Orange Juice",
            "Fresh Salsa", "Hummus Original", "Guacamole Fresh"
        ]
        
        self.grocery_produce_items = [
            "Bananas Bunch", "Apples Gala", "Apples Fuji", "Oranges Navel",
            "Strawberries 1lb", "Blueberries Pint", "Grapes Red Seedless", "Grapes Green",
            "Avocados", "Tomatoes Vine Ripe", "Lettuce Romaine", "Lettuce Iceberg",
            "Spinach Fresh", "Carrots 2lb Bag", "Celery Stalk", "Cucumbers",
            "Bell Peppers Red", "Bell Peppers Green", "Broccoli Crown", "Cauliflower",
            "Potatoes Russet 5lb", "Sweet Potatoes", "Onions Yellow", "Garlic Bulb"
        ]
        
        self.grocery_pantry_items = [
            "Bread White Sliced", "Bread Whole Wheat", "Tortillas Flour", "Bagels Plain",
            "Rice White 2lb", "Pasta Spaghetti", "Pasta Penne", "Cereal Cheerios",
            "Cereal Frosted Flakes", "Oatmeal Rolled", "Peanut Butter Creamy",
            "Jam Strawberry", "Olive Oil Extra Virgin", "Vegetable Oil",
            "Canned Tomatoes", "Canned Beans Black", "Canned Corn", "Canned Tuna",
            "Chicken Broth", "Pasta Sauce Marinara", "Ketchup", "Mustard",
            "Mayonnaise", "Salt", "Pepper Black", "Sugar White", "Flour All-Purpose",
            "Coffee Ground", "Tea Bags", "Crackers Saltine", "Chips Potato",
            "Cookies Chocolate Chip", "Granola Bars", "Bottled Water 24-Pack"
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
            # Smartphones & Tablets (Specific Models)
            "Apple iPhone 15 Pro 256GB", "Apple iPhone 15 128GB", "Apple iPhone 14 Pro Max",
            "Samsung Galaxy S24 Ultra", "Samsung Galaxy S24", "Samsung Galaxy Z Flip 5",
            "Google Pixel 8 Pro", "Google Pixel 8", "OnePlus 12 Pro",
            "Apple iPad Pro 12.9-inch", "Apple iPad Air 11-inch", "Samsung Galaxy Tab S9",
            # Laptops & Computers (Specific Models)
            "Apple MacBook Air M3 13-inch", "Apple MacBook Pro 14-inch M3", "Apple MacBook Pro 16-inch",
            "Dell XPS 13 Plus", "Dell XPS 15", "HP Spectre x360 14",
            "Lenovo ThinkPad X1 Carbon", "ASUS ROG Zephyrus G14", "Microsoft Surface Laptop 5",
            "Apple iMac 24-inch M3", "HP Pavilion Desktop", "Dell Inspiron Desktop",
            # Audio - Headphones & Earbuds (Specific Models)
            "Apple AirPods Pro 2nd Gen", "Apple AirPods 3rd Gen", "Apple AirPods Max",
            "Sony WH-1000XM5 Headphones", "Bose QuietComfort 45", "Bose QuietComfort Ultra Earbuds",
            "Samsung Galaxy Buds2 Pro", "Beats Studio Pro", "Beats Fit Pro",
            "JBL Tune 770NC Headphones", "Sennheiser Momentum 4", "Anker Soundcore Liberty 4",
            # Audio - Speakers (Specific Models)
            "Sonos Era 100 Speaker", "Bose SoundLink Flex", "JBL Flip 6",
            "JBL Charge 5", "Ultimate Ears Boom 3", "Amazon Echo Dot 5th Gen",
            "Apple HomePod mini", "Google Nest Audio", "Sonos Roam",
            # Smart Watches & Fitness Trackers (Specific Models)
            "Apple Watch Series 9 45mm", "Apple Watch SE 44mm", "Apple Watch Ultra 2",
            "Samsung Galaxy Watch 6", "Fitbit Charge 6", "Garmin Forerunner 265",
            "Google Pixel Watch 2", "Garmin Venu 3", "Amazfit GTR 4",
            # Cameras & Photography (Specific Models)
            "Sony Alpha a7 IV Camera", "Canon EOS R6 Mark II", "Nikon Z6 III",
            "GoPro HERO12 Black", "DJI Mini 4 Pro Drone", "Insta360 X3",
            "Elgato Ring Light", "Neewer RGB LED Light Panel", "Manfrotto Tripod",
            # Gaming (Specific Models)
            "PlayStation 5 Console", "Xbox Series X", "Nintendo Switch OLED",
            "Steam Deck 512GB", "PlayStation DualSense Controller", "Xbox Elite Controller",
            "Logitech G Pro Wireless Mouse", "Razer DeathAdder V3", "SteelSeries Rival 3",
            "Razer BlackWidow V4 Keyboard", "Logitech G915 TKL", "Corsair K70 RGB",
            # Cables & Chargers (Branded)
            "Anker USB-C to USB-C Cable", "Apple Lightning to USB-C Cable", "Belkin USB-C Cable 6ft",
            "Anker PowerLine HDMI Cable", "Amazon Basics Ethernet Cable",
            "Anker PowerPort III Charger", "Belkin 3-in-1 Wireless Charger", "Mophie Powerstation Plus",
            "Anker 737 Power Bank 24000mAh", "Belkin BoostCharge Pro",
            # Computer Accessories (Branded)
            "Logitech MX Master 3S Mouse", "Apple Magic Mouse", "Razer Basilisk V3",
            "Logitech MX Keys Keyboard", "Apple Magic Keyboard", "Keychron K8 Pro",
            "Logitech C920 Webcam", "Razer Kiyo Pro Webcam", "Elgato Facecam",
            "Samsung T7 Portable SSD 1TB", "SanDisk Extreme Pro 512GB", "Western Digital My Passport 2TB",
            "Kingston USB 3.2 Flash Drive 128GB", "SanDisk Ultra microSD 256GB",
            # Phone Accessories (Branded)
            "OtterBox Defender Case", "Spigen Ultra Hybrid Case", "Case-Mate Clear Case",
            "Apple MagSafe Charger", "Belkin Screen Protector", "PopSockets PopGrip",
            "iOttie Car Mount", "Spigen Phone Stand", "ESR Kickstand Case",
            # Smart Home (Specific Models)
            "Amazon Echo Show 8", "Google Nest Hub Max", "Ring Video Doorbell Pro 2",
            "Philips Hue Starter Kit", "TP-Link Kasa Smart Plug", "Wyze Cam v3",
            # Monitors & Displays (Specific Models)
            "Dell UltraSharp 27-inch Monitor", "LG UltraGear 32-inch Gaming Monitor",
            "Samsung Odyssey G7 Monitor", "BenQ PD3220U 4K Monitor",
            # Tablets & E-Readers
            "Amazon Kindle Paperwhite", "Amazon Fire HD 10", "Samsung Galaxy Tab A8",
            # Other Accessories
            "Apple Pencil 2nd Gen", "Wacom Intuos Pro Tablet", "Huion Graphics Tablet",
            "Logitech Keyboard Case for iPad", "Lamicall Tablet Stand",
            # TVs & Streaming (Specific Models)
            "Samsung 55-inch QLED 4K TV", "LG 65-inch OLED C3 TV", "Sony 75-inch Bravia X90L",
            "TCL 50-inch 4K Roku TV", "Hisense 43-inch Smart TV",
            "Apple TV 4K 128GB", "Roku Streaming Stick 4K", "Amazon Fire TV Stick 4K Max",
            "Google Chromecast with Google TV", "NVIDIA Shield TV Pro",
            # Printers & Scanners (Specific Models)
            "HP OfficeJet Pro 9025e", "Canon PIXMA TR8620", "Epson EcoTank ET-2850",
            "Brother HL-L2350DW Laser Printer", "HP LaserJet Pro M404n", "Fujitsu ScanSnap iX1600",
            # Networking & Connectivity (Specific Models)
            "TP-Link Archer AX55 WiFi 6 Router", "Netgear Nighthawk AX6000", "ASUS RT-AX86U",
            "Google Nest WiFi Pro 6E", "Eero Pro 6E Mesh System", "TP-Link Deco X55 Mesh",
            "Netgear Orbi RBK753 Mesh", "TP-Link WiFi Extender RE505X",
            # Security & Surveillance (Specific Models)
            "Arlo Pro 5 Security Camera", "Blink Outdoor 4 Camera", "eufy SoloCam S40",
            "Ring Stick Up Cam Battery", "Google Nest Cam Indoor", "Wyze Cam Pan v3",
            # E-Readers & Digital Paper
            "Kindle Oasis 32GB", "Kobo Libra 2", "reMarkable 2 Digital Paper Tablet"
        ]
        
        # Payment methods (PAYMENT_METHOD) - DEPRECATED: Use ProductionRandomizer.get_payment_method() instead
        # Kept for backward compatibility with old code paths
        self.payment_methods = [
            "Credit Card", "Debit Card", "Cash", "PayPal",
            "Apple Pay", "Google Pay", "Gift Card", "Venmo"
        ]
        
        # Category mappings for brand name generation
        self.category_map = {
            'fashion': 'fashion',
            'accessories': 'fashion',
            'jewelry': 'jewelry',
            'beauty': 'beauty',
            'home_garden': 'home_garden',
            'sports_fitness': 'sports_fitness',
            'pet_supplies': 'pet_supplies',
            'books_media': 'books_media',
            'toys_games': 'toys_games',
            'food_beverage': 'food_beverage',
            'health_wellness': 'health_wellness',
            'electronics': 'electronics',
            'wholesale_general': 'food_beverage',
            'wholesale_food': 'food_beverage',
            'wholesale_electronics': 'electronics',
            'wholesale_fashion': 'fashion',
            'distributor': 'food_beverage',
            'manufacturer': 'electronics',
            'dtc_fashion': 'fashion',
            'dtc_electronics': 'electronics',
            'dtc_beauty': 'beauty',
            'dtc_home': 'home_garden',
            'marketplace_handmade': 'fashion',
            'marketplace_vintage': 'fashion',
            'marketplace_artisan': 'fashion',
            'print_on_demand': 'fashion',
            'grocery_delivery': 'food_beverage',
            'meal_kit': 'food_beverage',
            'supplements': 'health_wellness',
            'plants': 'home_garden'
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
        elif category == 'grocery_delivery':
            # For grocery delivery, select from all zones to ensure variety
            all_grocery_items = (
                self.grocery_frozen_items +
                self.grocery_refrigerated_items +
                self.grocery_produce_items +
                self.grocery_pantry_items
            )
            description = random.choice(all_grocery_items)
            unit = random.choice(['ea', 'lb', 'oz', 'pkg'])
        elif category == 'health_wellness':
            description = random.choice(self.health_wellness_items)
            unit = random.choice(['ea', 'capsules', 'oz'])
        elif category == 'electronics':
            description = random.choice(self.electronics_items)
            unit = 'ea'
        else:
            # Default to fashion for unknown categories (no generic fallback)
            description = random.choice(self.fashion_items)
            unit = 'ea'
        
        # ITEM_QTY, ITEM_UNIT_COST, ITEM_TOTAL_COST
        # Use production-ready realistic quantities and prices
        quantity = ProductionRandomizer.get_realistic_quantity(category)
        unit_price = ProductionRandomizer.get_realistic_price(category, use_common=True)
        total = round(quantity * unit_price, 2)
        
        # ITEM_SKU (UPC)
        upc = self.fake.ean13()
        sku = self.fake.bothify(text='SKU-######')
        
        # ITEM_TAX
        tax_rate = random.choice([0.0, 6.5, 7.5, 8.25, 9.0])
        tax_amount = round(total * (tax_rate / 100), 2) if tax_rate > 0 else 0.0
        
        # ITEM_DISCOUNT (realistic seasonal patterns)
        current_month = datetime.now().month
        discount_pct, has_discount = ProductionRandomizer.get_seasonal_discount(current_month, category)
        discount = 0.0
        promotion = None
        if has_discount:
            discount = round(total * discount_pct, 2)
            promotion = f"{int(discount_pct*100)}% OFF"
        
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
            name=description,  # Set name field for inventory adjustment forms
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
    
    def add_fuel_station_fields(self, item: RetailLineItem, is_fuel: bool = True) -> RetailLineItem:
        """Add fuel station-specific fields to a line item"""
        if is_fuel:
            item.is_fuel = True
            item.fuel_grade = random.choice(['Regular', 'Mid-Grade', 'Premium', 'Diesel'])
            item.octane_rating = {'Regular': '87', 'Mid-Grade': '89', 'Premium': '93', 'Diesel': None}.get(item.fuel_grade)
            item.gallons = round(random.uniform(8.0, 20.0), 3)
            item.price_per_gallon = round(random.uniform(3.15, 4.89), 3)
            item.total = round(item.gallons * item.price_per_gallon, 2)
            item.unit_price = item.price_per_gallon
            item.quantity = item.gallons
            item.description = f"{item.fuel_grade} Unleaded" if item.fuel_grade != 'Diesel' else 'Diesel Fuel'
            
            # Rewards discount (30% chance)
            if random.random() < 0.3:
                item.rewards_discount = round(random.uniform(0.05, 0.20), 2)
                item.total_savings = round(item.gallons * item.rewards_discount, 2)
        return item
    
    def add_pharmacy_fields(self, item: RetailLineItem, is_prescription: bool = True) -> RetailLineItem:
        """Add pharmacy-specific fields to a line item"""
        if is_prescription:
            drugs = ['Lisinopril', 'Metformin', 'Atorvastatin', 'Omeprazole', 'Amlodipine', 
                     'Metoprolol', 'Simvastatin', 'Losartan', 'Gabapentin', 'Levothyroxine']
            item.rx_number = self.fake.bothify(text='RX#######')
            item.drug_name = random.choice(drugs)
            item.strength = random.choice(['5mg', '10mg', '20mg', '40mg', '50mg', '100mg', '500mg'])
            item.form = random.choice(['Tablets', 'Capsules', 'Liquid', 'Cream'])
            item.quantity = random.choice([30, 60, 90])
            item.unit = item.form
            item.prescriber = self.fake.last_name().upper()
            item.description = f"{item.drug_name} {item.strength} {item.form}"
            
            # Insurance coverage (70% chance)
            if random.random() < 0.7:
                item.insurance_covered = True
                retail_price = round(random.uniform(50.0, 250.0), 2)
                item.copay = round(random.uniform(5.0, 25.0), 2)
                item.insurance_paid = round(retail_price - item.copay, 2)
                item.insurance_savings = item.insurance_paid
                item.total = item.copay
            else:
                item.total = round(random.uniform(20.0, 80.0), 2)
            
            item.unit_price = round(item.total / item.quantity, 2)
            
            # Refills
            item.refills_remaining = random.randint(0, 5)
            refill_date = datetime.now() + timedelta(days=random.randint(20, 30))
            item.refill_after_date = refill_date.strftime('%Y-%m-%d')
        return item
    
    def add_wholesale_fields(self, item: RetailLineItem, store_type: str = 'wholesale_general') -> RetailLineItem:
        """Add B2B wholesale-specific fields to a line item"""
        # Case packaging
        item.units_per_case = random.choice([6, 12, 24, 36, 48, 72])
        item.case_quantity = random.randint(1, 20)
        item.price_per_case = round(item.unit_price * item.units_per_case * 0.85, 2)  # 15% bulk discount
        
        # MOQ and lead time
        item.moq = random.choice([1, 5, 10, 25, 50])
        item.lead_time = random.choice(['Same Day', '1-2 Days', '3-5 Days', '1-2 Weeks'])
        
        # Origin and manufacturer
        item.country_of_origin = random.choice(['USA', 'China', 'Mexico', 'Canada', 'Germany', 'Vietnam'])
        item.manufacturer = random.choice(['Acme Manufacturing', 'Global Brands Inc', 'Premier Supplies', 'Quality Goods Co'])
        
        # Volume discounts
        if item.case_quantity >= 10:
            item.volume_discount_percentage = random.choice([5.0, 7.5, 10.0, 12.5])
            item.volume_discount_amount = round((item.price_per_case * item.case_quantity) * (item.volume_discount_percentage / 100), 2)
        
        # Stock and availability
        item.stock_status = random.choice(['In Stock', 'Low Stock', 'Backordered', 'Pre-order'])
        if item.stock_status == 'In Stock':
            item.available_quantity = random.randint(100, 5000)
        
        # Calculate totals
        item.total_units = item.units_per_case * item.case_quantity
        item.quantity = item.total_units
        item.line_total = round(item.price_per_case * item.case_quantity, 2)
        if item.volume_discount_amount:
            item.line_total = round(item.line_total - item.volume_discount_amount, 2)
        item.total = item.line_total
        
        return item
    
    def add_ecommerce_variant_fields(self, item: RetailLineItem, category: str = 'fashion') -> RetailLineItem:
        """Add Shopify e-commerce variant fields (size, color, options) with realistic variation"""
        # Set the category on the item for Shopify product classification
        category_display_map = {
            'fashion': 'Apparel',
            'dtc_fashion': 'Apparel',
            'accessories': 'Accessories',
            'electronics': 'Electronics',
            'dtc_electronics': 'Electronics',
            'beauty': 'Beauty & Personal Care',
            'dtc_beauty': 'Beauty & Personal Care',
            'home_garden': 'Home & Garden',
            'dtc_home': 'Home & Garden',
            'jewelry': 'Jewelry',
            'sports_fitness': 'Sports & Fitness',
            'pet_supplies': 'Pet Supplies',
            'toys_games': 'Toys & Games',
            'food_beverage': 'Food & Beverage',
            'health_wellness': 'Health & Wellness',
            'books_media': 'Books & Media',
            'party_supplies': 'Party Supplies',
            'office_supplies': 'Office Supplies',
            'automotive': 'Automotive',
            'baby_kids': 'Baby & Kids',
            'outdoor': 'Outdoor & Recreation',
        }
        item.category = category_display_map.get(category, category.replace('_', ' ').title())
        
        if category in ['fashion', 'dtc_fashion', 'accessories']:
            # Set brand if not already set by add_fashion_fields()
            if not item.brand:
                fashion_brands = ['Nike', 'Adidas', 'Levi\'s', 'Gap', 'H&M', 'Zara', 'Uniqlo', 'Old Navy', 
                                 'American Eagle', 'Abercrombie', 'Hollister', 'Forever 21', 'Express', 
                                 'Banana Republic', 'J.Crew', 'Madewell', 'Urban Outfitters', 'Anthropologie']
                item.brand = random.choice(fashion_brands)
            
            # Use correlated realistic color/size combinations
            color, size = ProductionRandomizer.get_correlated_variant(category)
            if color and size:
                item.color = color
                item.size = size
            else:
                # Fallback to independent selection
                size_options = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '0', '2', '4', '6', '8', '10', '12', '14', '16', 'One Size']
                item.size = random.choice(size_options)
                colors = ['Black', 'White', 'Navy', 'Gray', 'Charcoal', 'Heather Gray', 'Red', 'Burgundy', 
                         'Blue', 'Royal Blue', 'Sky Blue', 'Green', 'Olive', 'Forest Green', 'Beige', 'Tan',
                         'Brown', 'Pink', 'Blush', 'Purple', 'Lavender', 'Yellow', 'Mustard', 'Orange', 'Coral']
                item.color = random.choice(colors)
            
            item.variant = f"{item.color} / {item.size}"
            materials = ['100% Cotton', '100% Organic Cotton', '95% Cotton 5% Spandex', 'Polyester Blend', 
                        '100% Linen', 'Cotton-Linen Blend', 'Denim', 'Raw Denim', 'Wool Blend', 
                        'Merino Wool', 'Cashmere Blend', 'Silk', 'Rayon', 'Modal', 'Bamboo Blend']
            item.material = random.choice(materials)
            # Add style tags
            item.style = random.choice(['Casual', 'Business Casual', 'Formal', 'Athletic', 'Streetwear', 'Vintage', 'Minimalist', 'Bohemian'])
            
        elif category in ['electronics', 'dtc_electronics']:
            brands = ['Apple', 'Samsung', 'Google', 'Sony', 'LG', 'HP', 'Dell', 'Lenovo', 'Asus', 'Anker', 'Bose', 'JBL', 'Logitech']
            item.brand = random.choice(brands)
            item.model_number = self.fake.bothify(text='??-####-??').upper()
            # Add color options for tech products
            tech_colors = ['Black', 'White', 'Silver', 'Space Gray', 'Gold', 'Rose Gold', 'Blue', 'Green', 'Red']
            item.color = random.choice(tech_colors)
            storage_options = ['64GB', '128GB', '256GB', '512GB', '1TB', '2TB']
            storage = random.choice(storage_options) if random.random() < 0.6 else None
            item.variant = f"{item.brand} {item.model_number}" + (f" - {item.color}" if item.color else "") + (f" - {storage}" if storage else "")
            item.warranty_period = random.choice(['1 Year', '2 Years', '3 Years'])
            item.warranty_months = int(item.warranty_period.split()[0]) * 12
            
        elif category in ['beauty', 'dtc_beauty']:
            # Set brand if not already set
            if not item.brand:
                beauty_brands = ['L\'Oreal', 'Maybelline', 'Revlon', 'CoverGirl', 'MAC', 'Urban Decay', 
                                'Too Faced', 'Clinique', 'Estee Lauder', 'Lancome', 'NARS', 'Fenty Beauty',
                                'Glossier', 'The Ordinary', 'Drunk Elephant', 'Tatcha', 'Sunday Riley']
                item.brand = random.choice(beauty_brands)
            
            # Beauty product shades
            shades = ['Fair', 'Light', 'Medium', 'Tan', 'Deep', 'Rose', 'Berry', 'Nude', 'Coral', 'Plum', 
                     'Mauve', 'Crimson', 'Ruby', 'Mocha', 'Espresso', 'Honey', 'Warm Beige', 'Cool Beige']
            item.color = random.choice(shades)
            sizes = ['5ml', '10ml', '15ml', '30ml', '50ml', '100ml', '0.17 oz', '0.5 oz', '1 oz', '2 oz', '3.4 oz']
            item.size = random.choice(sizes)
            item.variant = f"{item.color} / {item.size}"
            # Add finish type
            if random.random() < 0.5:
                item.finish = random.choice(['Matte', 'Glossy', 'Satin', 'Shimmer', 'Metallic', 'Natural'])
            
        elif category in ['home_garden', 'dtc_home']:
            # Set brand if not already set
            if not item.brand:
                home_brands = ['IKEA', 'West Elm', 'CB2', 'Pottery Barn', 'Crate & Barrel', 'Target',
                              'Threshold', 'Project 62', 'Room Essentials', 'Made Goods', 'Serena & Lily']
                item.brand = random.choice(home_brands)
            
            # Home decor colors
            home_colors = ['White', 'Off-White', 'Cream', 'Black', 'Charcoal', 'Gray', 'Light Gray', 'Beige', 
                          'Navy', 'Sage', 'Terracotta', 'Blush', 'Mustard', 'Teal', 'Natural', 'Walnut', 'Oak']
            item.color = random.choice(home_colors)
            dimensions = ['4x6"', '5x7"', '8x10"', '11x14"', '12x16"', '16x20"', '18x24"', '24x36"', 
                         '12x12"', '18x18"', '20x20"', '24x24"', 'Small', 'Medium', 'Large', 'X-Large']
            item.dimensions = random.choice(dimensions)
            item.variant = f"{item.color} / {item.dimensions}"
            materials = ['Wood', 'Solid Wood', 'Reclaimed Wood', 'Metal', 'Wrought Iron', 'Stainless Steel',
                        'Ceramic', 'Porcelain', 'Glass', 'Tempered Glass', 'Fabric', 'Linen', 'Cotton', 
                        'Velvet', 'Leather', 'Faux Leather', 'Marble', 'Concrete', 'Bamboo', 'Rattan']
            item.material = random.choice(materials)
        
        elif category in ['marketplace_handmade', 'marketplace_artisan']:
            # Handmade/artisan specific fields
            item.seller = self.fake.company()
            item.seller_rating = round(random.uniform(4.2, 5.0), 1)
            item.handmade = True
            item.made_to_order = random.random() < 0.3
            if item.made_to_order:
                item.lead_time = random.choice(['3-5 business days', '1-2 weeks', '2-3 weeks', '4-6 weeks'])
        
        # ===== TIER 1: HIGH IMPACT CATEGORIES =====
        elif category in ['sports_fitness', 'outdoor']:
            # Sports & Outdoor brands
            if not item.brand:
                sports_brands = ['Nike', 'Adidas', 'Under Armour', 'Reebok', 'Lululemon', 'Gymshark', 'Puma',
                                'The North Face', 'Patagonia', 'Columbia', 'REI', 'Yeti', 'Coleman']
                item.brand = random.choice(sports_brands)
            
            # Size and color
            size_options = ['XS', 'S', 'M', 'L', 'XL', '2XL', 'One Size', 'Small', 'Medium', 'Large', 'X-Large']
            item.size = random.choice(size_options)
            colors = ['Black', 'Navy', 'Gray', 'Red', 'Blue', 'Green', 'White', 'Orange', 'Purple', 'Yellow']
            item.color = random.choice(colors)
            item.variant = f"{item.color} / {item.size}"
            item.style = random.choice(['Athletic', 'Training', 'Running', 'Yoga', 'Hiking', 'Camping'])
            
        elif category == 'jewelry':
            # Jewelry brands
            if not item.brand:
                jewelry_brands = ['Tiffany & Co', 'Pandora', 'Swarovski', 'Kay Jewelers', 'Zales', 
                                 'Blue Nile', 'James Allen', 'Etsy Artisan', 'Cartier', 'David Yurman']
                item.brand = random.choice(jewelry_brands)
            
            # Material (using existing material field)
            materials = ['Sterling Silver', '14K Gold', '18K Gold', 'White Gold', 'Rose Gold', 'Platinum',
                        'Stainless Steel', 'Titanium', 'Brass', 'Copper', 'Mixed Metals']
            item.material = random.choice(materials)
            
            # Size (ring sizes, bracelet lengths, etc.)
            size_options = ['5', '6', '7', '8', '9', '10', '6.5', '7.5', '8.5', 
                           '16"', '18"', '20"', '24"', 'One Size', 'Adjustable']
            item.size = random.choice(size_options)
            item.variant = f"{item.material} / Size {item.size}"
            item.style = random.choice(['Classic', 'Modern', 'Vintage', 'Minimalist', 'Statement', 'Dainty'])
            
        elif category in ['toys_games']:
            # Toy brands
            if not item.brand:
                toy_brands = ['LEGO', 'Hasbro', 'Mattel', 'Fisher-Price', 'Melissa & Doug', 'Ravensburger',
                             'Hot Wheels', 'Barbie', 'Nerf', 'Play-Doh', 'Disney', 'Marvel']
                item.brand = random.choice(toy_brands)
            
            # Age range (using style field for now)
            age_ranges = ['Ages 3+', 'Ages 5+', 'Ages 8+', 'Ages 12+', 'Ages 6-12', 'Ages 3-5', 'All Ages']
            item.style = random.choice(age_ranges)
            
            # Color
            colors = ['Red', 'Blue', 'Green', 'Yellow', 'Pink', 'Purple', 'Orange', 'Multi-Color', 'Rainbow']
            item.color = random.choice(colors) if random.random() < 0.7 else None
            
            if item.color:
                item.variant = f"{item.color} / {item.style}"
            else:
                item.variant = item.style
                
        elif category == 'pet_supplies':
            # Pet brand
            if not item.brand:
                pet_brands = ['Blue Buffalo', 'Purina', 'Hill\'s Science Diet', 'Royal Canin', 'Kong', 
                             'Nylabone', 'Frontline', 'Seresto', 'Wellness', 'Taste of the Wild']
                item.brand = random.choice(pet_brands)
            
            # Size (weight or volume)
            sizes = ['5 lb', '15 lb', '30 lb', '40 lb', '8 oz', '12 oz', '24 oz', 'Small', 'Medium', 'Large']
            item.size = random.choice(sizes)
            
            # Flavor/color
            flavors = ['Chicken', 'Beef', 'Salmon', 'Turkey', 'Lamb', 'Duck', 'Peanut Butter', 
                      'Bacon', 'Cheese', 'Mixed', 'Natural']
            item.color = random.choice(flavors)  # Using color field for flavor
            item.variant = f"{item.size} / {item.color}"
            
        # ===== TIER 2: MEDIUM IMPACT CATEGORIES =====
        elif category in ['food_beverage', 'grocery_delivery', 'meal_kit']:
            # Food brands
            if not item.brand:
                food_brands = ['Coca-Cola', 'Pepsi', 'Kraft', 'Nestle', 'General Mills', 'Kellogg\'s',
                              'Lay\'s', 'Doritos', 'Pringles', 'Oreo', 'Hershey\'s', 'M&M\'s']
                item.brand = random.choice(food_brands)
            
            # Size
            sizes = ['12 oz', '16 oz', '20 oz', '1 lb', '2 lb', '5 lb', '500g', '1 kg', 
                    '6 pack', '12 pack', '24 pack', 'Single', 'Family Size']
            item.size = random.choice(sizes)
            
            # Flavor
            flavors = ['Original', 'Classic', 'Chocolate', 'Vanilla', 'Strawberry', 'BBQ', 'Sour Cream',
                      'Ranch', 'Spicy', 'Honey', 'Caramel', 'Mint', 'Cherry', 'Lemon']
            item.color = random.choice(flavors)  # Using color field for flavor
            item.variant = f"{item.size} / {item.color}"
            
        elif category == 'health_wellness':
            # Health brands
            if not item.brand:
                health_brands = ['Nature Made', 'Centrum', 'Vitamin Shoppe', 'GNC', 'NOW Foods', 
                                'Garden of Life', 'Optimum Nutrition', 'Quest', 'Pure Protein']
                item.brand = random.choice(health_brands)
            
            # Size (count or volume)
            sizes = ['30 ct', '60 ct', '90 ct', '120 ct', '180 ct', '8 oz', '16 oz', '32 oz', '2 lb', '5 lb']
            item.size = random.choice(sizes)
            
            # Strength (using style field)
            strengths = ['500mg', '1000mg', '1500mg', '100mg', '250mg', 'Regular Strength', 'Extra Strength']
            item.style = random.choice(strengths)
            item.variant = f"{item.size} / {item.style}"
            
        elif category == 'baby_kids':
            # Baby brands
            if not item.brand:
                baby_brands = ['Pampers', 'Huggies', 'Gerber', 'Fisher-Price', 'Carter\'s', 'Graco',
                              'Chicco', 'Britax', 'Skip Hop', 'Aden + Anais']
                item.brand = random.choice(baby_brands)
            
            # Size (ages or diaper sizes)
            sizes = ['Newborn', 'Size 1', 'Size 2', 'Size 3', 'Size 4', 'Size 5', 'Size 6',
                    '0-3M', '3-6M', '6-12M', '12-18M', '18-24M', '2T', '3T', '4T']
            item.size = random.choice(sizes)
            
            # Color
            colors = ['Blue', 'Pink', 'White', 'Neutral', 'Gray', 'Yellow', 'Green', 'Multi-Color']
            item.color = random.choice(colors)
            item.variant = f"{item.size} / {item.color}"
            
        # ===== TIER 3: LOWER IMPACT CATEGORIES =====
        elif category == 'books_media':
            # Publisher/brand
            if not item.brand:
                publishers = ['Penguin', 'HarperCollins', 'Simon & Schuster', 'Random House', 'Macmillan',
                             'Sony', 'Universal', 'Warner', 'HBO', 'Netflix']
                item.brand = random.choice(publishers)
            
            # Format (using style field)
            formats = ['Hardcover', 'Paperback', 'Mass Market', 'Kindle', 'Audiobook', 'Blu-ray', 'DVD', '4K Ultra HD']
            item.style = random.choice(formats)
            
            # Edition (using size field for edition info)
            editions = ['1st Edition', '2nd Edition', 'Collector\'s Edition', 'Special Edition', 'Standard']
            item.size = random.choice(editions) if random.random() < 0.4 else None
            
            if item.size:
                item.variant = f"{item.style} / {item.size}"
            else:
                item.variant = item.style
                
        elif category == 'party_supplies':
            # Party brands
            if not item.brand:
                party_brands = ['Amscan', 'Unique Industries', 'Creative Converting', 'Oriental Trading',
                               'Party City', 'Wilton', 'Balloon Time']
                item.brand = random.choice(party_brands)
            
            # Theme (using style field)
            themes = ['Birthday', 'Wedding', 'Baby Shower', 'Graduation', 'Halloween', 'Christmas',
                     'New Year', 'Summer', 'Princess', 'Superhero', 'Unicorn', 'Dinosaur']
            item.style = random.choice(themes)
            
            # Color
            colors = ['Blue', 'Pink', 'Red', 'Gold', 'Silver', 'White', 'Black', 'Rainbow', 'Pastel']
            item.color = random.choice(colors)
            item.variant = f"{item.color} / {item.style}"
            
        elif category == 'office_supplies':
            # Office brands
            if not item.brand:
                office_brands = ['Staples', 'Office Depot', 'Avery', 'Sharpie', '3M', 'BIC',
                                'Post-it', 'Scotch', 'Pilot', 'Paper Mate', 'Swingline']
                item.brand = random.choice(office_brands)
            
            # Size/quantity
            sizes = ['12 pack', '24 pack', '50 pack', '100 pack', 'Single', 'Box of 10', 'Box of 50',
                    '8.5x11"', '11x17"', 'Letter Size', 'Legal Size']
            item.size = random.choice(sizes)
            
            # Color
            colors = ['Black', 'Blue', 'Red', 'Green', 'Yellow', 'White', 'Assorted', 'Multi-Color']
            item.color = random.choice(colors)
            item.variant = f"{item.color} / {item.size}"
            
        elif category == 'automotive':
            # Auto brands
            if not item.brand:
                auto_brands = ['Mobil', 'Castrol', 'Bosch', 'ACDelco', 'Michelin', 'Goodyear',
                              'Pennzoil', 'Valvoline', 'Shell', 'Fram', 'NGK', 'Champion']
                item.brand = random.choice(auto_brands)
            
            # Viscosity/size (using size field)
            sizes = ['5W-20', '5W-30', '10W-30', '10W-40', '1 qt', '5 qt', '6 qt', '1 gal']
            item.size = random.choice(sizes)
            
            # Model compatibility (using style field)
            item.style = random.choice(['Universal', 'OEM', 'Performance', 'Synthetic', 'Conventional'])
            item.variant = f"{item.size} / {item.style}"
        
        # Add collection/line information (applies to all categories)
        if random.random() < 0.4:  # 40% of items have collection info
            current_year = datetime.now().year
            seasonal_collections = [
                f'Spring {current_year}', f'Summer {current_year}', f'Fall {current_year}', f'Winter {current_year}',
                f'Holiday {current_year}', f'Resort {current_year}', 'Back to School', 'Black Friday',
            ]
            series_collections = [
                'Pro Series', 'Premium Line', 'Deluxe Collection', 'Elite Series', 'Signature Collection',
                'Limited Edition', 'Essentials', 'Basics', 'Classic Collection', 'Modern Line',
                'Heritage Collection', 'Contemporary Series', 'Urban Collection', 'Sport Line'
            ]
            # Mix seasonal and series collections
            all_collections = seasonal_collections + series_collections
            item.collection = random.choice(all_collections)
        
        # Add product URL with realistic format
        product_slug = item.description.lower().replace(' ', '-').replace("'", '')
        item.product_url = f"https://shop.example.com/products/{product_slug}-{item.sku.lower()[:8]}"
        
        # Personalization options (more realistic patterns)
        if random.random() < 0.12:
            personalization_options = [
                'Monogram initials',
                'Gift wrap included',
                'Custom engraving',
                'Include gift message',
                'Rush processing',
                'Add gift receipt',
                'Special packaging'
            ]
            item.personalization = random.choice(personalization_options)
        
        # Add condition for marketplace items
        if 'marketplace' in category:
            item.condition = random.choices(
                ['New', 'New with tags', 'Like new', 'Excellent', 'Good', 'Fair'],
                weights=[60, 15, 10, 8, 5, 2]
            )[0]
        else:
            item.condition = 'New'
        
        return item
    
    def add_electronics_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add electronics-specific fields to a line item"""
        brands = ['Samsung', 'Sony', 'LG', 'Apple', 'Dell', 'HP', 'Lenovo', 'Canon', 'Nikon']
        item.brand = random.choice(brands)
        item.model_number = self.fake.bothify(text='??-####-???').upper()
        
        # Tech specs
        specs_templates = {
            'Laptop': {'Processor': 'Intel i7', 'RAM': '16GB', 'Storage': '512GB SSD', 'Display': '15.6"'},
            'TV': {'Screen Size': '55"', 'Resolution': '4K UHD', 'Refresh Rate': '120Hz', 'HDR': 'Yes'},
            'Phone': {'Storage': '256GB', 'RAM': '8GB', 'Camera': '48MP', 'Battery': '4500mAh'},
            'Camera': {'Megapixels': '24.2MP', 'Sensor': 'APS-C', 'ISO': '100-25600', 'Video': '4K'},
        }
        product_type = random.choice(list(specs_templates.keys()))
        item.tech_specs = specs_templates[product_type]
        item.configuration = ', '.join(f"{k}: {v}" for k, v in item.tech_specs.items())
        
        # Warranty
        item.warranty_period = random.choice(['1 Year', '2 Years', '3 Years'])
        warranty_date = datetime.now() + timedelta(days=365 * int(item.warranty_period[0]))
        item.warranty_expiry = warranty_date.strftime('%Y-%m-%d')
        
        # Extended warranty (40% chance)
        if random.random() < 0.4:
            item.extended_warranty = True
            item.extended_warranty_period = random.choice(['1 Year', '2 Years'])
        
        # Accessories (30% chance)
        if random.random() < 0.3:
            item.accessories = random.sample(['HDMI Cable', 'Carrying Case', 'Screen Protector', 
                                             'Wireless Mouse', 'USB-C Adapter', 'Memory Card'], 
                                            k=random.randint(1, 3))
        return item
    
    def add_fashion_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add fashion/apparel-specific fields to a line item"""
        item.brand = random.choice(['Nike', 'Adidas', 'Levi\'s', 'Gap', 'H&M', 'Zara', 'Uniqlo'])
        item.size = random.choice(['XS', 'S', 'M', 'L', 'XL', '2XL', '6', '8', '10', '12', '14'])
        item.color = random.choice(['Black', 'Navy', 'Gray', 'White', 'Red', 'Blue', 'Green'])
        item.material = random.choice(['100% Cotton', '65% Polyester 35% Cotton', 'Wool Blend', 'Denim'])
        item.style = random.choice(['Casual', 'Formal', 'Athletic', 'Relaxed', 'Slim Fit'])
        item.fit = random.choice(['Regular', 'Slim', 'Relaxed', 'Athletic'])
        item.care_instructions = 'Machine wash cold, tumble dry low'
        
        # Sale pricing (40% chance)
        if random.random() < 0.4:
            item.original_price = round(item.unit_price / 0.7, 2)  # 30% off
            item.savings = round(item.original_price - item.unit_price, 2)
            item.on_sale = True
        
        return item
    
    def add_grocery_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add grocery-specific fields to a line item"""
        item.brand = random.choice(['Organic Valley', 'Kirkland', 'Great Value', 'Store Brand'])
        item.organic = random.random() < 0.3
        item.locally_grown = random.random() < 0.2
        
        # Weight/unit pricing
        if random.random() < 0.5:
            item.weight_unit = random.choice(['lb', 'oz', 'kg'])
            item.weight = round(random.uniform(0.5, 5.0), 2)
            item.price_per_weight = round(item.unit_price / item.weight, 2)
        else:
            item.unit_measure = random.choice(['each', 'bunch', 'bag', 'box'])
            item.price_per_unit = item.unit_price
        
        # Expiration date
        exp_days = random.randint(3, 14)
        exp_date = datetime.now() + timedelta(days=exp_days)
        item.expiration_date = exp_date.strftime('%Y-%m-%d')
        
        # Sales and substitutions
        item.on_sale = random.random() < 0.3
        if random.random() < 0.1:
            item.substituted = True
            item.original_item = f"Original {item.description}"
        
        return item
    
    def add_home_improvement_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add home improvement-specific fields to a line item"""
        item.brand = random.choice(['DeWalt', 'Milwaukee', 'Ryobi', 'Black & Decker', 'Craftsman'])
        item.model_number = self.fake.bothify(text='???-####').upper()
        
        # Dimensions and measurements
        item.dimensions = f"{random.randint(12, 48)}x{random.randint(12, 48)} inches"
        item.finish = random.choice(['Matte', 'Gloss', 'Satin', 'Semi-Gloss'])
        item.color = random.choice(['White', 'Beige', 'Gray', 'Black', 'Wood Tone'])
        item.color_hex = random.choice(['#FFFFFF', '#F5F5DC', '#808080', '#000000', '#8B4513'])
        
        # Coverage/capacity
        if 'Paint' in item.description:
            item.capacity = '1 Gallon'
            item.coverage = 'Covers 400 sq ft'
            item.measure_unit = 'sq ft'
            item.price_per_measure = round(item.unit_price / 400, 4)
        
        # Installation and warranty
        item.installation_required = random.random() < 0.4
        item.warranty = random.choice(['90 Days', '1 Year', '2 Years', 'Lifetime'])
        item.material = random.choice(['Steel', 'Aluminum', 'Plastic', 'Wood', 'Composite'])
        
        return item
    
    def add_wholesale_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add wholesale/bulk-specific fields to a line item"""
        item.brand = random.choice(['Costco', 'Sam\'s Club', 'BJ\'s', 'Restaurant Depot'])
        item.manufacturer = random.choice(['Procter & Gamble', 'Unilever', 'PepsiCo', 'Nestlé'])
        item.country_of_origin = random.choice(['USA', 'Mexico', 'China', 'Canada'])
        
        # Bulk quantities
        item.units_per_case = random.choice([12, 24, 36, 48])
        item.case_quantity = random.randint(1, 10)
        item.total_units = item.units_per_case * item.case_quantity
        item.price_per_case = round(item.unit_price * item.units_per_case, 2)
        item.line_total = round(item.price_per_case * item.case_quantity, 2)
        item.total = item.line_total
        
        # Pallet quantities (20% chance)
        if random.random() < 0.2:
            item.cases_per_pallet = random.choice([40, 60, 80])
            item.pallet_quantity = 1
            item.price_per_pallet = round(item.price_per_case * item.cases_per_pallet, 2)
        
        # MOQ and lead time
        item.moq = random.choice([1, 5, 10, 20])
        item.lead_time = random.choice(['Same Day', '1-2 Days', '3-5 Days', '1 Week'])
        item.stock_status = random.choice(['In Stock', 'Low Stock', 'Available'])
        item.available_quantity = random.randint(50, 500)
        
        # Volume discounts
        if item.case_quantity >= 5:
            item.volume_discount_percentage = random.choice([5, 10, 15])
            item.volume_discount_amount = round(item.line_total * item.volume_discount_percentage / 100, 2)
        
        # Member/promotional discounts
        if random.random() < 0.4:
            item.member_discount = round(item.line_total * 0.05, 2)
        
        return item
    
    def add_digital_product_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add digital product-specific fields to a line item"""
        item.product_type = random.choice(['Software', 'Game', 'Subscription', 'E-book', 'Music'])
        item.platform = random.choice(['Windows', 'Mac', 'Multi-platform', 'iOS', 'Android'])
        item.version = random.choice(['2024', 'Pro', 'v3.5', 'Ultimate', 'Standard'])
        
        # Licensing
        item.license_type = random.choice(['Perpetual', 'Subscription', 'Single-user', 'Multi-user'])
        if 'Subscription' in item.license_type:
            item.subscription_period = random.choice(['1 Month', '3 Months', '1 Year'])
            renewal_date = datetime.now() + timedelta(days=30 if '1 Month' in item.subscription_period else 365)
            item.renewal_date = renewal_date.strftime('%Y-%m-%d')
            item.auto_renew = random.random() < 0.5
        
        item.license_seats = random.choice([1, 5, 10])
        item.license_key = self.fake.bothify(text='????-????-????-????').upper()
        item.activation_code = self.fake.bothify(text='########').upper()
        
        # Download/access
        item.download_url = f"https://download.{item.description.lower().replace(' ', '')}.com"
        item.access_url = f"https://app.{item.description.lower().replace(' ', '')}.com"
        item.file_size = f"{round(random.uniform(0.5, 5.0), 1)} GB"
        item.system_requirements = "Windows 10/11, 8GB RAM, 10GB disk space"
        
        # Expiry (if applicable)
        if random.random() < 0.3:
            expiry = datetime.now() + timedelta(days=random.randint(365, 1095))
            item.expiry_date = expiry.strftime('%Y-%m-%d')
        
        return item
    
    def add_qsr_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add quick service restaurant-specific fields to a line item"""
        # Modifiers (60% chance)
        if random.random() < 0.6:
            modifiers_pool = ['No pickles', 'Extra cheese', 'No onions', 'Add bacon', 
                             'Light mayo', 'Extra sauce', 'No lettuce', 'Well done']
            item.modifiers = random.sample(modifiers_pool, k=random.randint(1, 3))
        
        # Special instructions
        if random.random() < 0.3:
            item.special_instructions = random.choice([
                'Cut in half', 'No ice', 'Extra hot', 'On the side', 'Lightly toasted'
            ])
        
        # Combo meals (40% chance)
        if random.random() < 0.4:
            item.is_combo = True
            item.combo_savings = round(random.uniform(1.0, 3.0), 2)
            item.discount = item.combo_savings
        
        return item
    
    def add_marketplace_fields(self, item: RetailLineItem) -> RetailLineItem:
        """Add marketplace-specific fields to a line item"""
        item.seller_name = random.choice(['Best Deals Shop', 'Quality Goods LLC', 'Prime Sellers', 
                                          'Top Rated Store', 'Verified Vendor'])
        item.seller_sku = self.fake.bothify(text='SLKR-######')
        item.condition = random.choice(['New', 'Used - Like New', 'Used - Very Good', 'Used - Good', 'Refurbished'])
        
        # Variant (60% chance)
        if random.random() < 0.6:
            variants = [
                'Color: Blue, Size: M',
                'Size: Large',
                'Color: Black',
                'Style: Modern',
                '256GB, Midnight'
            ]
            item.variant = random.choice(variants)
        
        return item
    
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
        
        # Merchant information - use dynamic brand name generation for infinite realistic names
        category = self.category_map.get(store_type, 'fashion')
        receipt.supplier_name = BrandNameGenerator.generate_brand_name(store_type, style='auto')
        receipt.supplier_address = self.fake.address().replace('\n', ', ')
        receipt.supplier_phone = self.fake.phone_number()
        receipt.supplier_email = self.fake.company_email()
        receipt.store_website = f"www.{receipt.supplier_name.lower().replace(' ', '').replace('&', 'and')[:50]}.com"
        
        # Generate logo for supplier
        receipt.supplier_logo = self.logo_generator.generate_logo(
            company_name=receipt.supplier_name,
            category=category,
            style='auto',
            size=(200, 80)
        )
        
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
        
        # Generate line items with specialized fields based on store type
        num_items = random.randint(min_items, max_items)
        
        # Determine if this is a specialized store type that needs custom fields
        specialized_types = {
            'fuel': 'add_fuel_station_fields',
            'pharmacy': 'add_pharmacy_fields', 
            'qsr': 'add_qsr_fields'
        }
        
        for i in range(num_items):
            item = self.generate_line_item(category=store_type)
            
            # Add specialized fields based on store type
            if store_type == 'fuel':
                # For fuel stations: first item is usually fuel, rest are convenience items
                if i == 0:
                    item = self.add_fuel_station_fields(item, is_fuel=True)
                else:
                    item = self.add_fuel_station_fields(item, is_fuel=False)
            elif store_type == 'pharmacy':
                # For pharmacies: 60% prescriptions, 40% OTC items
                if random.random() < 0.6:
                    item = self.add_pharmacy_fields(item, is_prescription=True)
                else:
                    item = self.add_pharmacy_fields(item, is_prescription=False)
            elif store_type == 'qsr':
                item = self.add_qsr_fields(item)
            
            receipt.line_items.append(item)
        
        # Calculate totals (round to avoid floating point precision errors like 959.5600000000001)
        receipt.subtotal = round(sum(item.total for item in receipt.line_items), 2)  # SUBTOTAL
        receipt.discount = round(sum(item.discount for item in receipt.line_items), 2)  # DISCOUNT
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
        
        # PAYMENT_METHOD - use POS context (includes cash for retail)
        receipt.payment_method, payment_metadata = ProductionRandomizer.get_payment_method(context='pos')
        
        # PAYMENT_TERMS (card details, approval codes)
        if 'card_type' in payment_metadata:
            receipt.card_type = payment_metadata['card_type']
            receipt.card_last_four = payment_metadata['card_last_four']
            receipt.approval_code = payment_metadata.get('approval_code', self.fake.bothify(text='AUTH######'))
            receipt.transaction_id = self.fake.bothify(text='TXN##########')
            receipt.payment_terms = f"Card ending in {receipt.card_last_four}, Auth: {receipt.approval_code}"
        elif receipt.payment_method == "Cash":
            receipt.cash_tendered = round(receipt.total_amount + random.uniform(0, 20), 2)
            receipt.change_amount = round(receipt.cash_tendered - receipt.total_amount, 2)
            receipt.payment_terms = f"Cash tendered: ${receipt.cash_tendered:.2f}"
        elif receipt.payment_method == "Gift Card":
            receipt.card_last_four = payment_metadata.get('card_last_four', str(random.randint(1000, 9999)))
            receipt.payment_terms = f"Gift card ending in {receipt.card_last_four}"
        elif 'email_partial' in payment_metadata:
            # Digital wallets (Apple Pay, Google Pay)
            receipt.email_partial = payment_metadata['email_partial']
            receipt.approval_code = self.fake.bothify(text='AUTH######')
            receipt.payment_terms = f"Paid via {receipt.payment_method}, Auth: {receipt.approval_code}"
        else:
            receipt.payment_terms = receipt.payment_method
        
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
                             locale: Optional[str] = None,
                             mixed_categories: Optional[List[str]] = None) -> RetailReceiptData:
        """Generate an online order/invoice with ALL 37 entities + specialized fields"""
        
        # First generate the base receipt
        receipt = RetailReceiptData()
        
        # Copy the base receipt generation logic but add specialized fields
        if locale is None:
            locale_weights = {
                'en_US': 0.40, 'en_GB': 0.15, 'en_CA': 0.10, 'en_AU': 0.08,
                'fr_CA': 0.07, 'fr_FR': 0.05, 'es_ES': 0.05, 'es_MX': 0.04,
                'de_DE': 0.04, 'zh_CN': 0.02
            }
            locales = list(locale_weights.keys())
            weights = list(locale_weights.values())
            locale = random.choices(locales, weights=weights)[0]
        
        receipt.locale = locale
        
        # Override for online orders
        receipt.doc_type = "Invoice"
        receipt.invoice_number = self.fake.bothify(text='ORD-######')
        
        # ORDER_DATE (for online orders) - keep in ISO format
        order_date = self.fake.date_between(start_date='-30d', end_date='today')
        receipt.order_date = order_date.strftime('%Y-%m-%d')
        receipt.invoice_date = (order_date + timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d')
        
        # Merchant information - use dynamic brand name generation for infinite realistic names
        category = self.category_map.get(store_type, 'fashion')
        receipt.supplier_name = BrandNameGenerator.generate_brand_name(store_type, style='auto')
        receipt.supplier_address = self.fake.address().replace('\n', ', ')
        receipt.supplier_phone = self.fake.phone_number()
        receipt.supplier_email = self.fake.company_email()
        receipt.store_website = f"www.{receipt.supplier_name.lower().replace(' ', '').replace('&', 'and')[:50]}.com"
        
        # Generate logo for supplier
        receipt.supplier_logo = self.logo_generator.generate_logo(
            company_name=receipt.supplier_name,
            category=category,
            style='auto',
            size=(200, 80)
        )
        
        # Marketplace seller info (order-level) - always populate for online orders
        seller_names = ['Best Deals Shop', 'Quality Goods LLC', 'Prime Sellers', 
                       'Top Rated Store', 'Verified Vendor', 'TrustedSeller', 
                       'FastShip Pro', 'ValueMart', 'DirectGoods']
        receipt.seller_name = random.choice(seller_names)
        receipt.seller_username = self.fake.user_name()
        receipt.seller_rating = round(random.uniform(4.0, 5.0), 1)
        receipt.seller_reviews = random.randint(100, 50000)
        receipt.marketplace_name = random.choice(['Amazon', 'eBay', 'Etsy', 'Walmart', 'Target', 'Shopify'])
        
        # Customer information (REQUIRED for online orders)
        receipt.buyer_name = self.fake.name()
        receipt.buyer_address = self.fake.address().replace('\n', ', ')
        receipt.buyer_phone = self.fake.phone_number()
        receipt.buyer_email = self.fake.email()
        receipt.account_number = self.fake.bothify(text='ACCT######')
        receipt.customer_id = receipt.account_number
        
        # TRACKING_NUMBER (for online orders)
        receipt.tracking_number = self.fake.bothify(text='1Z###??########')
        
        # Generate line items with specialized fields
        num_items = random.randint(min_items, max_items)
        
        # Use mixed categories if provided, otherwise use single store_type
        if mixed_categories:
            # Ensure we have enough categories for all items
            if len(mixed_categories) < num_items:
                # Repeat categories to match num_items
                categories_to_use = mixed_categories * (num_items // len(mixed_categories) + 1)
                categories_to_use = categories_to_use[:num_items]
            else:
                categories_to_use = mixed_categories[:num_items]
        else:
            # Use single category for all items (backward compatibility)
            categories_to_use = [store_type] * num_items
        
        for category in categories_to_use:
            item = self.generate_line_item(category=category)
            
            # Add specialized fields based on category
            if category in ['electronics', 'wholesale_electronics']:
                item = self.add_electronics_fields(item)
            elif category in ['fashion', 'accessories', 'wholesale_fashion', 'marketplace_handmade', 'marketplace_vintage']:
                item = self.add_fashion_fields(item)
            elif category in ['food_beverage', 'wholesale_food', 'grocery_delivery', 'meal_kit']:
                item = self.add_grocery_fields(item)
            elif category in ['home_garden', 'plants']:
                item = self.add_home_improvement_fields(item)
            elif 'wholesale' in category:
                item = self.add_wholesale_fields(item)
            elif 'digital' in category or 'software' in category:
                item = self.add_digital_product_fields(item)
            
            # ✅ ADD SHOPIFY E-COMMERCE VARIANT FIELDS (category, variant, collection)
            # This populates the 11 Shopify-critical fields for all online orders
            item = self.add_ecommerce_variant_fields(item, category=category)

            
            # For marketplace orders, add seller info to items
            # Always add for marketplace templates, 30% chance otherwise
            if random.random() < 0.3:
                item = self.add_marketplace_fields(item)
            
            # Ensure item has seller_name (use order-level seller if not set)
            if not item.seller_name:
                item.seller_name = receipt.seller_name
            
            receipt.line_items.append(item)
        
        # Calculate totals (round to avoid floating point precision errors like 959.5600000000001)
        receipt.subtotal = round(sum(item.total for item in receipt.line_items), 2)
        receipt.discount = round(sum(item.discount for item in receipt.line_items), 2)
        receipt.total_discount = receipt.discount

        receipt.tax_rate = random.choice([6.5, 7.5, 8.0, 8.25, 9.0])
        taxable_amount = receipt.subtotal - receipt.discount
        receipt.tax_amount = round(taxable_amount * (receipt.tax_rate / 100), 2)
        
        # Add shipping cost for online orders
        if receipt.subtotal >= 50:
            # Free shipping over $50
            receipt.shipping_cost = 0.0
            receipt.shipping_method = "Free Standard Shipping"
        else:
            # Standard shipping rates based on order value
            receipt.shipping_cost = round(random.uniform(5.99, 12.99), 2)
            receipt.shipping_method = random.choice(["Standard Shipping", "Ground Shipping", "Economy Shipping"])
        
        receipt.total_amount = round(receipt.subtotal - receipt.discount + receipt.tax_amount + receipt.shipping_cost, 2)
        receipt.currency = "$"
        
        # Payment - use e-commerce context (NO CASH for online orders)
        receipt.payment_method, payment_metadata = ProductionRandomizer.get_payment_method(context='ecommerce')
        
        # Apply payment metadata
        if 'card_type' in payment_metadata:
            receipt.card_type = payment_metadata['card_type']
            receipt.card_last_four = payment_metadata['card_last_four']
            receipt.approval_code = payment_metadata.get('approval_code', self.fake.bothify(text='AUTH######'))
            receipt.transaction_id = self.fake.bothify(text='TXN##########')
            receipt.payment_terms = f"{receipt.card_type} ending in {receipt.card_last_four}"
        elif 'email_partial' in payment_metadata:
            receipt.email_partial = payment_metadata['email_partial']
            receipt.payment_terms = f"Paid via {receipt.payment_method} ({payment_metadata['email_partial']})"
        elif 'installments' in payment_metadata:
            # BNPL services (Afterpay, Klarna)
            installments = payment_metadata['installments']
            installment_amount = round(receipt.total_amount / installments, 2)
            receipt.payment_terms = f"{receipt.payment_method} - {installments} payments of ${installment_amount:.2f}"
        else:
            # Other payment methods
            receipt.payment_terms = receipt.payment_method
        
        receipt.has_table = True
        receipt.terms_and_conditions = "All sales final. No refunds on sale items."
        receipt.note = "Thank you for your order!"
        
        # Remove POS-specific fields
        receipt.register_number = None
        receipt.cashier_id = None
        receipt.transaction_time = None
        
        return receipt
    
    def _build_attributes_string(self, item: RetailLineItem) -> Optional[str]:
        """Build a formatted attributes string from specialized fields for display in templates.
        
        Examples:
        - Fashion: "Size: L, Color: Blue, Material: Cotton"
        - Electronics: "Brand: Sony, Model: WH-1000XM4, Warranty: 2 years"
        - Grocery: "Weight: 1.2 lbs, Organic, Expires: 2024-03-15"
        """
        attributes = []
        
        # Fashion attributes
        if item.size:
            attributes.append(f"Size: {item.size}")
        if item.color:
            attributes.append(f"Color: {item.color}")
        if item.material:
            attributes.append(f"Material: {item.material}")
        if item.fit:
            attributes.append(f"Fit: {item.fit}")
        
        # Electronics attributes
        if item.brand:
            attributes.append(f"Brand: {item.brand}")
        if item.model_number:
            attributes.append(f"Model: {item.model_number}")
        if item.warranty_period:
            attributes.append(f"Warranty: {item.warranty_period}")
        
        # Grocery attributes
        if item.organic:
            attributes.append("Organic")
        if item.weight and item.weight_unit:
            attributes.append(f"Weight: {item.weight} {item.weight_unit}")
        if item.expiration_date:
            attributes.append(f"Expires: {item.expiration_date}")
        
        # Fuel attributes
        if item.fuel_grade:
            attributes.append(f"Grade: {item.fuel_grade}")
        if item.gallons:
            attributes.append(f"{item.gallons} gal")
        
        # Pharmacy attributes
        if item.drug_name:
            attributes.append(f"Drug: {item.drug_name}")
        if item.strength:
            attributes.append(f"Strength: {item.strength}")
        if item.form:
            attributes.append(f"Form: {item.form}")
        
        # Home improvement attributes
        if item.dimensions:
            attributes.append(f"Dimensions: {item.dimensions}")
        if item.finish:
            attributes.append(f"Finish: {item.finish}")
        
        # Digital product attributes
        if item.platform:
            attributes.append(f"Platform: {item.platform}")
        if item.license_type:
            attributes.append(f"License: {item.license_type}")
        if item.subscription_period:
            attributes.append(f"Subscription: {item.subscription_period}")
        
        # Marketplace attributes
        if item.seller_name:
            attributes.append(f"Seller: {item.seller_name}")
        if item.condition and item.condition != 'New':
            attributes.append(f"Condition: {item.condition}")
        if item.variant:
            attributes.append(f"Variant: {item.variant}")
        
        # QSR attributes
        if item.modifiers:
            attributes.append(f"Modifiers: {item.modifiers}")
        if item.special_instructions:
            attributes.append(f"Notes: {item.special_instructions}")
        
        # Wholesale attributes
        if item.units_per_case and item.case_quantity:
            attributes.append(f"{item.case_quantity} cases × {item.units_per_case} units")
        
        return ", ".join(attributes) if attributes else None
    
    def generate_purchase_order(
        self,
        po_type: str = 'domestic',
        product_category: Optional[str] = None,
        min_items: int = 5,
        max_items: int = 20,
        moq_required: bool = False,
        payment_terms: str = 'NET 30',
        incoterms: Optional[str] = None,
        include_customization: bool = False,
    ):
        """
        Generate a realistic B2B purchase order.
        
        Args:
            po_type: Type of PO (alibaba, dropship, domestic, manufacturer, fashion, 
                     electronics, food_beverage, beauty, home_goods)
            product_category: Product category (auto-selected if None)
            min_items: Minimum line items
            max_items: Maximum line items
            moq_required: Include MOQ requirements
            payment_terms: Payment terms string
            incoterms: International shipping terms
            include_customization: Add custom specs
            
        Returns:
            PurchaseOrder instance
        """
        
        # Map po_type string to POType enum
        po_type_map = {
            'alibaba': POType.ALIBABA,
            'dropship': POType.DROPSHIP,
            'domestic': POType.DOMESTIC,
            'domestic_distributor': POType.DOMESTIC,
            'manufacturer': POType.MANUFACTURER,
            'manufacturer_direct': POType.MANUFACTURER,
            'fashion': POType.FASHION,
            'fashion_wholesale': POType.FASHION,
            'electronics': POType.ELECTRONICS,
            'food_beverage': POType.FOOD_BEVERAGE,
            'food': POType.FOOD_BEVERAGE,
            'beauty': POType.BEAUTY,
            'beauty_cosmetics': POType.BEAUTY,
            'home_goods': POType.HOME_GOODS,
            'home': POType.HOME_GOODS,
            'generic': POType.GENERIC,
        }
        
        # Handle both string and enum input
        if isinstance(po_type, POType):
            po_type_enum = po_type
        else:
            po_type_enum = po_type_map.get(po_type.lower(), POType.GENERIC)
        
        # Auto-select product category if not specified
        if not product_category:
            category_map = {
                POType.ALIBABA: 'wholesale_goods',
                POType.DROPSHIP: 'electronics',
                POType.DOMESTIC: 'wholesale_goods',
                POType.MANUFACTURER: 'electronics',
                POType.FASHION: 'fashion',
                POType.ELECTRONICS: 'electronics',
                POType.FOOD_BEVERAGE: 'food_beverage',
                POType.BEAUTY: 'health_wellness',
                POType.HOME_GOODS: 'home_garden',
                POType.GENERIC: 'wholesale_goods',
            }
            product_category = category_map[po_type_enum]
        
        # Generate buyer info (your company)
        buyer_info = self._generate_buyer_company()
        
        # Generate supplier info
        supplier_info = self._generate_supplier_company(po_type_enum, product_category)
        
        # Generate line items
        item_count = random.randint(min_items, max_items)
        line_items = []
        
        for i in range(item_count):
            item = self._generate_po_line_item(
                line_number=i + 1,
                po_type=po_type_enum,
                category=product_category,
                moq_required=moq_required
            )
            line_items.append(item)
        
        # Calculate delivery date
        lead_time = random.randint(14, 90)  # 14-90 days
        delivery_date = date.today() + timedelta(days=lead_time)
        
        # Determine incoterms for international POs
        if po_type_enum == POType.ALIBABA and not incoterms:
            incoterms = random.choice(['FOB Shanghai', 'CIF Los Angeles', 'EXW Shenzhen'])
        
        # Create purchase order
        po = PurchaseOrder(
            po_number=f"PO-{random.randint(2024, 2025)}-{random.randint(10000, 99999)}",
            po_date=date.today(),
            status=random.choice([POStatus.DRAFT, POStatus.SENT, POStatus.CONFIRMED]),
            
            buyer_company=buyer_info['company'],
            buyer_address=buyer_info['address'],
            buyer_contact=buyer_info['contact'],
            buyer_email=buyer_info['email'],
            buyer_phone=buyer_info['phone'],
            buyer_logo=buyer_info.get('logo'),
            
            supplier_name=supplier_info['name'],
            supplier_address=supplier_info['address'],
            supplier_contact=supplier_info['contact'],
            supplier_email=supplier_info['email'],
            supplier_phone=supplier_info['phone'],
            supplier_account_number=supplier_info.get('account_number'),
            
            ship_to_company=buyer_info['company'],
            ship_to_address=buyer_info['warehouse_address'],
            ship_to_attention="Receiving Department",
            ship_to_phone=buyer_info['phone'],
            
            line_items=line_items,
            
            shipping_cost=Decimal(str(random.uniform(50, 500))),
            tax_rate=Decimal(str(random.choice([0.0, 0.06, 0.07, 0.08, 0.09, 0.10]))),
            
            payment_terms=payment_terms,
            shipping_method=self._get_shipping_method(po_type_enum),
            incoterms=incoterms,
            delivery_date=delivery_date,
            
            special_instructions=self._get_po_instructions(po_type_enum),
            
            currency='USD',
            po_type=po_type_enum,
            created_by=buyer_info['contact'],
        )
        
        return po
    
    def _generate_buyer_company(self) -> Dict:
        """Generate buyer (your company) information"""
        company_names = [
            "E-Commerce Solutions Inc.",
            "Digital Retail Group",
            "Online Marketplace LLC",
            "Modern Commerce Co.",
            "Direct-to-Consumer Brands",
            "Omnichannel Retail Corp.",
            "Digital First Trading",
            "Smart Retail Systems",
            "NextGen E-Commerce",
            "Cloud Commerce Partners",
        ]
        
        company = random.choice(company_names)
        
        # Generate logo for buyer company
        buyer_logo = self.logo_generator.generate_logo(
            company_name=company,
            category='retail',
            style='auto',
            size=(200, 80)
        )
        
        return {
            'company': company,
            'address': self.fake.address().replace('\n', ', '),
            'warehouse_address': f"{self.fake.building_number()} Warehouse Drive, {self.fake.city()}, {self.fake.state_abbr()} {self.fake.zipcode()}",
            'contact': self.fake.name(),
            'email': self.fake.company_email(),
            'phone': self.fake.phone_number(),
            'logo': buyer_logo,
        }
    
    def _generate_supplier_company(self, po_type: 'POType', category: str) -> Dict:
        """Generate supplier/vendor information"""
        
        supplier_names = {
            POType.ALIBABA: [
                'Shenzhen Tech Manufacturing Co., Ltd.',
                'Guangzhou Wholesale Trading Company',
                'Yiwu Import Export Corporation',
                'Hangzhou Electronics Factory',
                'Shanghai Global Supply Chain Co.',
                'Dongguan Industrial Manufacturers',
                'Ningbo Ocean Trading Ltd.',
                'Foshan Production Group',
            ],
            POType.DROPSHIP: [
                'National Fulfillment Services',
                'Direct Ship Wholesale',
                'Rapid Fulfillment Partners',
                'Elite Drop Services',
                'Premier Distribution Network',
                'Express Fulfillment Solutions',
                'Streamline Drop Ship',
            ],
            POType.DOMESTIC: [
                'ABC Wholesale Distributors',
                'Midwest Supply Company',
                'National Trade Partners',
                'Regional Distribution Center',
                'Premier Wholesale Group',
                'United Distributors Inc.',
                'Continental Supply Co.',
            ],
            POType.MANUFACTURER: [
                'Custom Manufacturing Solutions',
                'Precision Products Inc.',
                'Quality Factory Direct',
                'Industrial Manufacturing Corp.',
                'Advanced Production Facilities',
                'Elite Manufacturing Group',
            ],
            POType.FASHION: [
                'Fashion Forward Wholesale',
                'Apparel Collective',
                'Style Source International',
                'Garment District Imports',
                'Trend Line Distributors',
                'Designer Wholesale Network',
            ],
        }
        
        # Get supplier name list for this PO type
        name_list = supplier_names.get(po_type, [
            'Global Wholesale Distributors',
            'International Trading Company',
            'Premium Supply Partners',
            'Worldwide Distribution Group',
        ])
        
        supplier_name = random.choice(name_list)
        
        # Generate appropriate address
        if po_type == POType.ALIBABA:
            cities = ['Shenzhen', 'Guangzhou', 'Shanghai', 'Hangzhou', 'Ningbo']
            city = random.choice(cities)
            address = f"{random.randint(1, 999)} Industrial Road, {city}, Guangdong, China"
        else:
            address = self.fake.address().replace('\n', ', ')
        
        return {
            'name': supplier_name,
            'address': address,
            'contact': self.fake.name(),
            'email': self.fake.company_email(),
            'phone': self.fake.phone_number(),
            'account_number': f"ACCT-{random.randint(10000, 99999)}" if random.random() > 0.3 else None,
        }
    
    def _generate_po_line_item(
        self,
        line_number: int,
        po_type: 'POType',
        category: str,
        moq_required: bool
    ):
        """Generate a single purchase order line item"""
        
        # Get product based on category - use simple product names
        products = {
            'fashion': [
                "Premium Cotton T-Shirt", "Denim Jeans Classic Fit", "Pullover Hoodie",
                "Athletic Performance Shorts", "Casual Button-Down Shirt", "Leather Belt",
                "Winter Jacket Insulated", "Yoga Pants Stretch", "Running Shoes",
            ],
            'electronics': [
                "Wireless Bluetooth Headphones", "USB-C Charging Cable", "Portable Power Bank",
                "Laptop Stand Adjustable", "Wireless Mouse", "LED Desk Lamp",
                "Smart Watch Fitness Tracker", "Phone Case Protective", "Screen Protector",
            ],
            'food_beverage': [
                "Organic Coffee Beans 1lb", "Green Tea Bags 100ct", "Protein Bar Box 12pk",
                "Almond Butter 16oz", "Olive Oil Extra Virgin 500ml", "Honey Raw 16oz",
                "Pasta Whole Wheat 1lb", "Canned Tomatoes 28oz", "Quinoa Organic 2lb",
            ],
            'health_wellness': [
                "Vitamin D3 Softgels 100ct", "Multivitamin Daily 60ct", "Fish Oil Omega-3",
                "Protein Powder Vanilla 2lb", "Face Cream Moisturizer", "Hand Sanitizer 8oz",
                "Body Lotion Unscented 16oz", "Sunscreen SPF 50", "Essential Oil Set",
            ],
            'home_garden': [
                "Throw Pillow Decorative", "Kitchen Knife Set 6pc", "Bath Towel Set 4pk",
                "Picture Frame 8x10", "Plant Pot Ceramic", "Storage Bins 3pk",
                "LED Light Bulbs 4pk", "Door Mat Welcome", "Wall Clock Modern",
            ],
            'wholesale_goods': [
                "Bulk Paper Towels 24pk", "Industrial Cleaner Gallon", "Office Pens 50ct",
                "Disposable Cups 100ct", "Plastic Bags 500ct", "Cardboard Boxes 25pk",
                "Packing Tape 6pk", "Labels Adhesive 1000ct", "Trash Bags 100ct",
            ],
        }
        
        product_list = products.get(category, products['wholesale_goods'])
        product = random.choice(product_list)
        
        # Generate SKUs
        supplier_sku = f"{random.choice(['SKU', 'PART', 'ITEM', 'PROD'])}-{random.randint(1000, 9999)}"
        # Dropship ALWAYS requires buyer_sku for SKU mapping
        if po_type == POType.DROPSHIP:
            buyer_sku = f"PROD-{random.randint(100, 999)}"
        else:
            buyer_sku = f"PROD-{random.randint(100, 999)}" if random.random() > 0.5 else None
        
        # Determine quantity and MOQ
        if moq_required or po_type == POType.ALIBABA:
            moq = random.choice([25, 50, 100, 144, 500, 1000])
            quantity = moq * random.randint(1, 5)
        else:
            moq = None
            quantity = random.randint(10, 500)
        
        # Pricing based on category
        price_ranges = {
            'electronics': (15, 500),
            'fashion': (5, 80),
            'food_beverage': (2, 25),
            'health_wellness': (3, 40),
            'home_garden': (10, 300),
            'wholesale_goods': (8, 150),
        }
        price_range = price_ranges.get(category, (5, 100))
        unit_cost = Decimal(str(round(random.uniform(*price_range), 2)))
        
        # Lead time
        lead_time_ranges = {
            POType.ALIBABA: (30, 90),
            POType.MANUFACTURER: (45, 120),
            POType.FASHION: (45, 90),
            POType.DOMESTIC: (7, 30),
            POType.DROPSHIP: (2, 14),
        }
        lead_range = lead_time_ranges.get(po_type, (14, 60))
        lead_time = random.randint(*lead_range)
        
        # Brand name
        brand_gen = BrandNameGenerator()
        brand = brand_gen.generate_brand_name(category)
        
        # HS Code for international trade (Alibaba)
        hs_code = None
        if po_type == POType.ALIBABA:
            hs_code = self._generate_hs_code(category, product)
        
        return PurchaseOrderLineItem(
            line_number=line_number,
            supplier_sku=supplier_sku,
            buyer_sku=buyer_sku,
            product_name=product,
            quantity_ordered=quantity,
            moq=moq,
            unit_of_measure=random.choice(['EA', 'CASE', 'BOX', 'CTN']),
            unit_cost=unit_cost,
            lead_time_days=lead_time,
            product_category=category,
            brand=brand,
            weight=round(random.uniform(0.5, 50.0), 2) if random.random() > 0.5 else None,
            hs_code=hs_code,
        )
    
    def _get_shipping_method(self, po_type: 'POType') -> str:
        """Get appropriate shipping method for PO type"""
        
        methods = {
            POType.ALIBABA: ['Ocean Freight', 'Air Freight', 'Express Air'],
            POType.DOMESTIC: ['Standard Ground', 'LTL Freight', 'FTL Freight', 'UPS Ground'],
            POType.DROPSHIP: ['Standard Shipping', 'Express Shipping', '2-Day Air'],
            POType.MANUFACTURER: ['Direct Truck', 'LTL Freight', 'Factory Pickup'],
        }
        return random.choice(methods.get(po_type, ['Standard Shipping']))
    
    def _generate_hs_code(self, category: str, product_name: str) -> str:
        """
        Generate realistic HS (Harmonized System) code for international trade.
        
        HS codes are 6-10 digits identifying product types for customs.
        Format: XXXX.XX.XXXX (Chapter.Heading.Subheading)
        
        Args:
            category: Product category
            product_name: Product name
            
        Returns:
            Formatted HS code string
        """
        # Common HS code prefixes by category
        hs_prefixes = {
            'fashion': [
                '6109',  # T-shirts, singlets, and other vests, knitted
                '6203',  # Men's suits, jackets, trousers
                '6204',  # Women's suits, jackets, trousers
                '6211',  # Tracksuits, ski suits, swimwear
                '6402',  # Footwear with rubber/plastic outer soles
                '4203',  # Articles of apparel of leather
            ],
            'electronics': [
                '8517',  # Telephone sets, cell phones, other apparatus
                '8471',  # Computers and peripheral equipment
                '8528',  # Monitors, projectors, televisions
                '8518',  # Microphones, speakers, headphones
                '8504',  # Power supply units, chargers, adapters
                '8523',  # Memory cards, USB drives
            ],
            'food_beverage': [
                '0901',  # Coffee, roasted or not
                '0902',  # Tea, whether or not flavored
                '2106',  # Food preparations (protein bars, supplements)
                '1905',  # Bread, pastry, cakes, biscuits
                '2009',  # Fruit juices
            ],
            'health_wellness': [
                '2106',  # Food supplements
                '3004',  # Medicaments (vitamins, supplements)
                '3305',  # Hair preparations
                '3307',  # Perfumes and toilet preparations
            ],
            'home_garden': [
                '6302',  # Bed, table, toilet, kitchen linens
                '8211',  # Knives with cutting blades
                '9403',  # Furniture
                '6304',  # Furnishing articles (cushions, curtains)
                '3924',  # Tableware, kitchenware of plastic
            ],
            'wholesale_goods': [
                '4819',  # Cartons, boxes, cases of paper/paperboard
                '3924',  # Household articles of plastic
                '8205',  # Hand tools
                '9608',  # Pens, pencils, markers
            ],
        }
        
        prefix_list = hs_prefixes.get(category, ['9999'])  # Generic code if unknown
        prefix = random.choice(prefix_list)
        
        # Generate subheading (2 digits) and item code (4 digits)
        subheading = random.randint(10, 99)
        item_code = random.randint(1000, 9999)
        
        return f"{prefix}.{subheading}.{item_code}"
    
    def _get_po_instructions(self, po_type: 'POType') -> Optional[str]:
        """Get special instructions for PO type"""
        
        instructions = {
            POType.ALIBABA: [
                "Please arrange pre-shipment inspection before shipping.",
                "Notify before shipping. Provide commercial invoice and packing list.",
                "Pack items in export cartons. Label with PO number.",
            ],
            POType.DROPSHIP: [
                "Use plain packaging - no supplier branding.",
                "Include our return address on all shipments.",
                "Email tracking numbers within 24 hours of ship.",
            ],
            POType.DOMESTIC: [
                "Call before delivery. Dock hours 8AM-4PM weekdays.",
                "Pack items in original manufacturer boxes.",
                "Palletize if total weight exceeds 500 lbs.",
            ],
        }
        
        inst_list = instructions.get(po_type)
        return random.choice(inst_list) if inst_list else None
    
    def _organize_grocery_items_by_zone(self, line_items: List[RetailLineItem]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize grocery line items by temperature zone for the grocery template"""
        zones = {
            'frozen_items': [],
            'refrigerated_items': [],
            'pantry_items': [],
            'produce_items': []
        }
        
        for item in line_items:
            # Assign zone based on product description
            desc_lower = item.description.lower()
            
            if any(word in desc_lower for word in ['ice cream', 'frozen', 'popsicle', 'freezer']):
                zone = 'frozen_items'
            elif any(word in desc_lower for word in ['milk', 'cheese', 'yogurt', 'dairy', 'meat', 'chicken', 'beef', 'pork', 'fish', 'refrigerated']):
                zone = 'refrigerated_items'
            elif any(word in desc_lower for word in ['apple', 'banana', 'orange', 'lettuce', 'tomato', 'vegetable', 'fruit', 'produce']):
                zone = 'produce_items'
            else:
                zone = 'pantry_items'  # Default: dry goods, canned, packaged
            
            # Convert item to dict with all fields
            item_dict = {
                'description': item.description,
                'quantity': item.quantity,
                'unit_price': item.unit_price,
                'total': item.total,
                'brand': getattr(item, 'brand', None),
                'size': getattr(item, 'size', None),
                'upc': item.upc,
                'weight': getattr(item, 'weight', None),
                'weight_unit': getattr(item, 'weight_unit', None),
                'expiration_date': getattr(item, 'expiration_date', None),
                'organic': getattr(item, 'organic', False),
                'on_sale': getattr(item, 'on_sale', False),
                'substituted': getattr(item, 'substituted', False),
                'original_item': getattr(item, 'original_item', None),
                'unit_measure': getattr(item, 'unit_measure', None),
                'price_per_unit': getattr(item, 'price_per_unit', None),
            }
            
            zones[zone].append(item_dict)
        
        return zones
    
    def to_dict(self, receipt: RetailReceiptData) -> Dict[str, Any]:
        """Convert RetailReceiptData to dictionary for template rendering"""
        base_dict = {
            # Locale for formatting
            'locale': receipt.locale,
            
            # Document metadata
            'doc_type': receipt.doc_type,
            'document_type': receipt.doc_type,  # Alias for templates
            'invoice_number': receipt.invoice_number,
            'invoice_date': receipt.invoice_date,
            'order_date': receipt.order_date,
            
            # Merchant
            'supplier_name': receipt.supplier_name,
            'supplier_logo': receipt.supplier_logo,
            'supplier_address': receipt.supplier_address,
            'supplier_phone': receipt.supplier_phone,
            'supplier_email': receipt.supplier_email,
            'store_website': receipt.store_website,
            # Premium receipt template aliases
            'supplier_initials': ''.join(word[0].upper() for word in receipt.supplier_name.split()[:2]) if receipt.supplier_name else '',
            'store_location': receipt.supplier_address,
            # Wholesale receipt template aliases
            'store_name': receipt.supplier_name,
            'store_address': receipt.supplier_address.split(',')[0] if receipt.supplier_address and ',' in receipt.supplier_address else receipt.supplier_address,
            'street_address': receipt.supplier_address.split(',')[0] if receipt.supplier_address and ',' in receipt.supplier_address else receipt.supplier_address,  # Alias for store_address
            'city': receipt.supplier_address.split(',')[1].strip() if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 1 else '',
            'state': receipt.supplier_address.split(',')[2].split()[0] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 else '',
            'zip_code': receipt.supplier_address.split(',')[2].split()[1] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 and len(receipt.supplier_address.split(',')[2].split()) > 1 else '',
            'postcode': receipt.supplier_address.split(',')[2].split()[1] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 and len(receipt.supplier_address.split(',')[2].split()) > 1 else '',  # UK/AU alias for zip_code
            'address_line_2': '',  # Second address line (usually empty for most stores)
            'address_line_3': '',  # Third address line
            'building_name': '',  # Building name (optional)
            'area_name': random.choice(['Downtown', 'Midtown', 'Uptown', 'West End', 'East Side', 'Central', 'Plaza District']),
            'industrial_area': random.choice(['Industrial Park', 'Commerce Center', 'Business District', 'Trade Zone']),
            'industrial_zone': random.choice(['Zone A', 'Zone B', 'Industrial Estate', 'Commercial Area']),
            'branch_location': random.choice(['Main Branch', 'Downtown', 'Mall Location', 'Airport', 'Suburb']),
            'phone': receipt.supplier_phone,
            'store_phone': receipt.supplier_phone,  # Alias
            'store_phone_alt': receipt.supplier_phone,  # Alternative phone
            'store_fax': f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",  # Fax number
            'business_hours': random.choice(['Mon-Sat 9AM-9PM, Sun 10AM-6PM', '24 Hours', 'Mon-Fri 8AM-8PM', 'Daily 7AM-10PM']),
            'store_hours': random.choice(['9:00 AM - 9:00 PM', '24 Hours', '8:00 AM - 10:00 PM', '7:00 AM - 11:00 PM']),
            'website': receipt.store_website,
            'store_number': f"{random.randint(100, 999)}",  # Generated store ID
            'terminal_id': f"T{random.randint(1, 20):02d}",  # Terminal/POS ID
            'terminal_number': f"{random.randint(1, 99):02d}",
            'counter_no': random.randint(1, 15),
            'operator_id': f"OP{random.randint(100, 999)}",
            'operator_number': f"{random.randint(1000, 9999)}",
            'tagline': random.choice([
                "Quality You Can Trust",
                "Your Satisfaction, Our Priority", 
                "Serving You Since 1985",
                "Shop Smart, Save Big",
                "Everyday Low Prices",
                "Where Quality Meets Value",
                "Excellence in Every Purchase",
                "Your One-Stop Shop"
            ]),  # Store slogan
            'store_tagline': random.choice([
                "Quality You Can Trust",
                "Your Satisfaction, Our Priority", 
                "Serving You Since 1985",
                "Shop Smart, Save Big",
                "Everyday Low Prices",
                "Where Quality Meets Value"
            ]),  # Alias for tagline
            
            # International tax/business IDs (common in Asia-Pacific receipts)
            'gst_id': f"GST{random.randint(100000000, 999999999)}",  # Goods & Services Tax ID
            'gst_reg': f"GST{random.randint(100000000, 999999999)}",  # GST registration alias
            'sst_id': f"SST{random.randint(10000000, 99999999)}",  # Sales & Services Tax ID (Malaysia)
            'company_reg': f"REG-{random.randint(100000, 999999)}",  # Company registration number
            'legal_suffix': random.choice(['Sdn Bhd', 'Pte Ltd', 'Pty Ltd', 'Ltd', 'Inc', 'LLC', 'Corp']),
            'tax_id': f"TAX-{random.randint(10000000, 99999999)}",  # Generic tax ID
            'resale_certificate': f"RC-{random.randint(1000000, 9999999)}",  # Resale certificate for wholesale
            
            # Warehouse/Wholesale fields
            'warehouse_location': random.choice(['Warehouse A', 'Distribution Center', 'Main Warehouse', 'Regional Hub']),
            'warehouse_code': f"WH{random.randint(100, 999)}",
            'warehouse_number': f"{random.randint(1, 50)}",
            'company_name': receipt.supplier_name,  # Alias for supplier_name
            
            # GST/Tax calculation fields (for Asian receipts)
            'amount_excluding_gst': f"${receipt.subtotal:.2f}",  # Subtotal before GST
            'total_quantity': sum(item.quantity for item in receipt.line_items) if receipt.line_items else 0,
            'total_units': sum(item.quantity for item in receipt.line_items) if receipt.line_items else 0,
            
            # Credit/Payment terms (B2B)
            'credit_limit': f"${random.randint(5000, 50000):.2f}",
            'credit_terms': random.choice(['Net 30', 'Net 15', 'Net 60', '2/10 Net 30']),
            'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'late_fee': f"${random.uniform(15, 50):.2f}",
            'outstanding_balance': f"${random.uniform(0, 5000):.2f}",
            
            # Rounding (common in cash transactions)
            'rounding_adjustment': f"${random.choice([0.01, 0.02, -0.01, -0.02, 0.00]):.2f}",
            
            # Receipt barcode (for scan/lookup)
            'receipt_barcode': f"{random.randint(1000000000000, 9999999999999)}",
            'barcode_number': f"{random.randint(1000000000000, 9999999999999)}",
            
            # Supplier city/state/zip (extracted from address)
            'supplier_city': receipt.supplier_address.split(',')[1].strip() if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 1 else '',
            'supplier_state': receipt.supplier_address.split(',')[2].split()[0] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 else '',
            'supplier_zip': receipt.supplier_address.split(',')[2].split()[1] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 and len(receipt.supplier_address.split(',')[2].split()) > 1 else '',
            'store_city': receipt.supplier_address.split(',')[1].strip() if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 1 else '',
            'store_state': receipt.supplier_address.split(',')[2].split()[0] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 else '',
            'store_zip': receipt.supplier_address.split(',')[2].split()[1] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 and len(receipt.supplier_address.split(',')[2].split()) > 1 else '',
            
            # Misc specialized fields
            'batch_number': f"BATCH-{random.randint(1000, 9999)}",
            'lot_number': f"LOT{random.randint(10000, 99999)}",
            'warranty_terms': '90-day warranty on parts and labor',
            'fuel_savings_total': f"${random.uniform(1, 10):.2f}",
            'free_delivery_minimum': f"${random.choice([35, 50, 75, 100]):.2f}",
            'business_center_phone': self.fake.phone_number(),
            'modifier': random.choice(['No onions', 'Extra cheese', 'Light ice', 'No salt', None]),
            
            # Additional address/location fields
            'street_name': receipt.supplier_address.split(',')[0].split()[-1] if receipt.supplier_address else 'Main St',
            'invoice_prefix': random.choice(['INV', 'ORD', 'REC', 'TXN']),
            'unit_code': f"U{random.randint(100, 999)}",
            
            # Electronics protection plans
            'protection_plan_name': random.choice(['Extended Protection', 'Complete Care', 'Total Guard', 'Premium Protection']),
            'protection_plan_price': f"${random.uniform(29.99, 199.99):.2f}",
            'protection_plan_coverage': random.choice(['2 years', '3 years', '4 years']),
            'protection_plan_duration': random.choice(['24 months', '36 months', '48 months']),
            'installation_fee': f"${random.uniform(49.99, 199.99):.2f}",
            'spec_key': random.choice(['Display', 'Processor', 'Storage', 'Memory']),
            'spec_value': random.choice(['15.6" FHD', 'Intel i7', '512GB SSD', '16GB DDR4']),
            'accessory': random.choice(['Carrying Case', 'Screen Protector', 'Charger', 'Mouse']),
            
            # Fashion
            'size_chart_url': f"https://sizechart.{self.fake.domain_name()}",
            
            # Service invoice fields
            'service_type': random.choice(['Repair', 'Installation', 'Maintenance', 'Inspection']),
            'service_category': random.choice(['HVAC', 'Plumbing', 'Electrical', 'Appliance']),
            'service_date': datetime.now().strftime('%B %d, %Y'),
            'service_location': receipt.buyer_address,
            'service_guarantee': '30-day service guarantee',
            'technician_name': f"{self.fake.first_name()} {self.fake.last_name()}",
            'license_number': f"LIC-{random.randint(100000, 999999)}",
            'labor_charges': f"${random.uniform(50, 200):.2f}",
            'materials_total': f"${random.uniform(25, 150):.2f}",
            'materials_subtotal': f"${random.uniform(25, 150):.2f}",
            'trip_charge': f"${random.uniform(25, 75):.2f}",
            'warranty_info': 'Parts: 1 year, Labor: 90 days',
            'appointment_time': random.choice(['9:00 AM - 11:00 AM', '11:00 AM - 1:00 PM', '2:00 PM - 4:00 PM']),
            'amount_paid': f"${receipt.total_amount:.2f}",
            'balance_due': f"$0.00",
            'special_instructions': random.choice(['Ring doorbell', 'Call upon arrival', 'Use side entrance', '']),
            'discount_description': 'First-time customer discount' if receipt.discount > 0 else None,
            'service': random.choice(['Annual Maintenance', 'Emergency Repair', 'Installation', 'Inspection']),
            'material': random.choice(['Replacement parts', 'Filters', 'Wiring', 'Fixtures']),
            
            # Home improvement specific
            'project_name': random.choice(['Kitchen Remodel', 'Bathroom Update', 'Deck Installation', 'Flooring']),
            'project_description': 'Project materials and installation',
            'installation_date': (datetime.now() + timedelta(days=random.randint(3, 14))).strftime('%B %d, %Y'),
            'installation_time_window': random.choice(['Morning (8AM-12PM)', 'Afternoon (12PM-5PM)', 'All Day']),
            'installation_total': f"${random.uniform(200, 1000):.2f}",
            'haul_away_fee': f"${random.uniform(25, 100):.2f}",
            'rental': random.choice(['Floor Sander', 'Tile Saw', 'Pressure Washer', None]),
            'rental_total': f"${random.uniform(50, 150):.2f}" if random.random() < 0.3 else None,
            'cash_back_earned': f"${random.uniform(5, 50):.2f}",
            'category': random.choice(['Tools', 'Lumber', 'Paint', 'Flooring', 'Plumbing']),
            'delivery_charge': f"${random.uniform(29, 99):.2f}",
            
            # Grocery specific
            'shopper_name': f"{self.fake.first_name()} {self.fake.last_name()[0]}.",
            'shopper_tip': f"${random.uniform(3, 15):.2f}",
            'items_subtotal': f"${receipt.subtotal:.2f}",
            'sub': f"${receipt.subtotal:.2f}",  # Alias for subtotal
            'free_delivery_threshold': f"${random.choice([35, 50, 75]):.2f}",
            
            # E-wallet fields
            'ewallet_type': random.choice(['Apple Pay', 'Google Pay', 'Samsung Pay', 'PayPal', None]),
            'ewallet_ref': f"EW{random.randint(100000, 999999)}" if random.random() < 0.3 else None,
            
            # Customer
            'buyer_name': receipt.buyer_name,
            'buyer_address': receipt.buyer_address,
            'buyer_phone': receipt.buyer_phone,
            'buyer_email': receipt.buyer_email,
            'customer_id': receipt.customer_id,
            'customer_name': receipt.buyer_name,  # Alias for buyer_name (used by premium/trading templates)
            'customer_company': receipt.buyer_name,  # Company name for B2B
            'customer_phone': receipt.buyer_phone,  # Alias for buyer_phone
            'customer_address': receipt.buyer_address,  # Alias for buyer_address
            'customer_gst': f"GST{random.randint(100000000, 999999999)}",  # Customer GST ID for B2B
            'customer_reg': f"REG{random.randint(100000, 999999)}",  # Customer registration
            'account_number': receipt.account_number,
            
            # Membership/Loyalty fields (used by wholesale, bookstore, dollar store)
            'member_id': receipt.customer_id or f"M{random.randint(100000, 999999)}",
            'member_name': receipt.buyer_name,
            'member_number': receipt.customer_id or f"{random.randint(1000000000, 9999999999)}",
            'member_tier': random.choice(['Gold', 'Silver', 'Bronze', 'Platinum', 'Standard', 'Premium', 'Executive']),
            'member_type': random.choice(['Business', 'Individual', 'Executive', 'Premium', 'Standard']),
            'tier': random.choice(['Gold', 'Silver', 'Bronze', 'Platinum', 'Standard']),  # Alias for member_tier
            'points_earned': random.randint(10, 500),
            'points_balance': random.randint(1000, 50000),
            'points_to_reward': random.randint(50, 500),  # Points needed for next reward
            'next_reward_value': f"${random.randint(5, 25)}",  # Value of next reward
            'next_reward_threshold': random.randint(1000, 5000),  # Points threshold for next reward
            'reward_progress': f"{random.randint(50, 95)}%",  # Progress to next reward
            'points_expiry': (datetime.now() + timedelta(days=365)).strftime('%B %d, %Y'),
            'member_promo': random.choice(['10% off next purchase', 'Double points week', 'Free shipping', None]),
            'rewards_earned': random.randint(5, 100),
            'rewards_balance': random.randint(100, 5000),
            'promotion_message': random.choice(['Buy 2 Get 1 Free!', 'Weekend Special!', 'Member Exclusive!', None]),
            'promo_message': random.choice(['Special Offer!', 'Limited Time!', 'Best Value!', None]),
            'promo': random.choice(['SAVE10', 'WEEKEND', 'MEMBER', None]),
            'exchange_policy': 'Items may be exchanged within 14 days with receipt',
            
            # Financial totals (RAW numeric values for templates that format themselves)
            'currency': receipt.currency,
            'currency_symbol': receipt.currency,  # Alias for templates
            'subtotal': receipt.subtotal,  # RAW numeric
            'subtotal_raw': receipt.subtotal,  # For calculations
            'sub_total': receipt.subtotal,  # RAW numeric alias
            'tax_amount': receipt.tax_amount if receipt.tax_amount else 0.0,  # RAW numeric
            'tax_amount_raw': receipt.tax_amount,  # For calculations
            'tax': receipt.tax_amount if receipt.tax_amount else 0.0,  # RAW numeric alias
            'sales_tax': receipt.tax_amount if receipt.tax_amount else 0.0,  # RAW numeric
            'tax_rate': receipt.tax_rate,
            'total_amount': receipt.total_amount,  # RAW numeric
            'total_amount_raw': receipt.total_amount,  # For calculations
            'total': receipt.total_amount,  # RAW numeric alias
            'discount': receipt.discount if receipt.discount > 0 else 0.0,  # RAW numeric
            'discount_raw': receipt.discount,  # For calculations
            'total_discount': receipt.total_discount if receipt.total_discount > 0 else 0.0,  # RAW numeric
            'total_discount_raw': receipt.total_discount,  # For calculations
            'total_savings': f"${receipt.total_discount:.2f}" if receipt.total_discount > 0 else (f"${receipt.discount:.2f}" if receipt.discount > 0 else None),  # Alias for total_discount
            
            # Coupon fields (for online orders and retail)
            'coupon_code': f"SAVE{random.randint(10, 50)}" if receipt.discount > 0 else None,
            'coupon_discount': f"${receipt.discount:.2f}" if receipt.discount > 0 else None,
            'coupon_savings': f"${receipt.discount:.2f}" if receipt.discount > 0 else None,
            'loyalty_discount': f"${receipt.discount * 0.5:.2f}" if receipt.discount > 0 else None,
            'bulk_discount': f"${receipt.discount:.2f}" if receipt.discount > 0 else None,
            'trade_discount': f"${receipt.discount:.2f}" if receipt.discount > 0 else None,
            'member_discount_amount': f"${receipt.discount:.2f}" if receipt.discount > 0 else None,
            'member_total_savings': f"${receipt.total_discount:.2f}" if receipt.total_discount > 0 else None,
            'savings_percentage': f"{random.randint(5, 25)}%",
            
            # Cashback fields (wholesale clubs)
            'cashback_earned': f"${random.uniform(1, 20):.2f}",
            'cashback_balance': f"${random.uniform(10, 200):.2f}",
            
            # Bottle deposit / bag charge (environmental fees)
            'bottle_deposit': f"${random.choice([0.05, 0.10, 0.00]):.2f}",
            'bag_charge': f"${random.choice([0.05, 0.10, 0.00]):.2f}",
            'bag_count': random.randint(0, 5),
            
            'tip_amount': receipt.tip_amount if receipt.tip_amount > 0 else 0.0,  # RAW numeric
            'tip_amount_raw': receipt.tip_amount,  # For calculations
            'shipping_cost': receipt.shipping_cost if hasattr(receipt, 'shipping_cost') and receipt.shipping_cost > 0 else 0.0,  # RAW numeric
            'shipping_cost_raw': receipt.shipping_cost if hasattr(receipt, 'shipping_cost') else 0.0,  # For calculations
            'shipping': receipt.shipping_cost if hasattr(receipt, 'shipping_cost') else 0.0,  # RAW numeric alias
            'shipping_fee': receipt.shipping_cost if hasattr(receipt, 'shipping_cost') else 0.0,  # RAW numeric alias
            
            # Item count
            'item_count': len(receipt.line_items),
            'total_items': sum(item.quantity for item in receipt.line_items) if receipt.line_items else 0,  # Total quantity of all items
            'num_items': len(receipt.line_items),  # Number of line items
            
            # Payment
            'payment_method': receipt.payment_method,
            'payment_terms': receipt.payment_terms,
            'card_type': receipt.card_type,
            'card_last_four': receipt.card_last_four,
            'approval_code': receipt.approval_code,
            'transaction_id': receipt.transaction_id,
            'cash_tendered': receipt.cash_tendered if receipt.cash_tendered else 0.0,  # RAW numeric
            'amount_tendered': receipt.cash_tendered if receipt.cash_tendered else 0.0,  # RAW numeric alias
            'change_amount': receipt.change_amount if receipt.change_amount else 0.0,  # RAW numeric
            'change_due': receipt.change_amount if receipt.change_amount else 0.0,  # RAW numeric alias
            
            # Fuel station fields (top-level)
            'pump_number': random.randint(1, 16),
            'station_number': f"ST{random.randint(100, 999)}",
            'odometer_reading': random.randint(10000, 200000),
            'vehicle_id': f"VEH-{random.randint(1000, 9999)}",
            'driver_id': f"DRV-{random.randint(100, 999)}",
            
            # Staff fields
            'store_manager': f"{self.fake.first_name()} {self.fake.last_name()[0]}.",
            'manager_name': f"{self.fake.first_name()} {self.fake.last_name()[0]}.",
            'salesperson': f"{self.fake.first_name()} {self.fake.last_name()[0]}.",
            'sales_rep': f"{self.fake.first_name()} {self.fake.last_name()}",
            
            # QSR / Restaurant fields
            'order_type': random.choice(['Dine In', 'Take Out', 'Drive Thru', 'Delivery', 'Pickup']),
            'franchise_number': f"FR{random.randint(1000, 9999)}",
            'delivery_driver': f"{self.fake.first_name()} {self.fake.last_name()[0]}.",
            
            # Line items (with all specialized fields preserved)
            'line_items': [
                {
                    # Base fields
                    'description': item.description,
                    'name': item.name or item.description,  # Alias
                    'quantity': item.quantity,
                    'unit_price': item.unit_price,  # RAW numeric value for templates that format themselves
                    'total': item.total,  # RAW numeric value for templates that format themselves
                    'upc': item.upc,
                    'sku': item.sku,
                    'unit': item.unit,
                    'tax_rate': item.tax_rate,
                    'tax_amount': item.tax_amount if item.tax_amount else 0.0,  # RAW numeric
                    
                    # Legacy field aliases for backward compatibility
                    'qty': item.quantity,  # Alias for quantity
                    'rate': item.unit_price,  # RAW numeric (some templates format themselves)
                    'price': item.unit_price,  # RAW numeric
                    'unit_cost': item.unit_price,  # RAW numeric
                    'amount': item.total,  # RAW numeric
                    'total_price': item.total,  # RAW numeric (wholesale template)
                    'discount': item.discount if item.discount > 0 else 0.0,  # RAW numeric
                    'discount_amount': item.discount if item.discount > 0 else 0.0,  # RAW numeric
                    'discount_percent': f"{int((item.discount / item.total * 100) if item.total > 0 and item.discount > 0 else 0)}%",
                    
                    # Item identification aliases
                    'barcode': item.upc,  # Alias for upc
                    'item_code': item.sku or item.upc,  # Product code
                    'part_number': item.sku,  # Auto parts alias
                    'isbn': f"978-{random.randint(0,9)}-{random.randint(10000,99999)}-{random.randint(100,999)}-{random.randint(0,9)}" if random.random() < 0.3 else None,  # Book ISBN
                    'batch_code': f"B{random.randint(1000, 9999)}",
                    'promotion_code': f"PROMO{random.randint(100, 999)}" if item.promotion else None,
                    
                    'lot_number': item.lot_number,
                    'serial_number': item.serial_number,
                    'weight': item.weight,
                    'promotion': item.promotion,
                    'rewards_earned': item.rewards_earned,
                    
                    # Fuel station fields
                    'is_fuel': item.is_fuel,
                    'fuel_grade': item.fuel_grade,
                    'octane_rating': item.octane_rating,
                    'gallons': item.gallons,
                    'price_per_gallon': f"${item.price_per_gallon:.3f}" if item.price_per_gallon else None,
                    'rewards_discount': f"${item.rewards_discount:.2f}" if item.rewards_discount else None,
                    'total_savings': f"${item.total_savings:.2f}" if item.total_savings else None,
                    
                    # Pharmacy fields
                    'rx_number': item.rx_number,
                    'drug_name': item.drug_name,
                    'strength': item.strength,
                    'form': item.form,
                    'prescriber': item.prescriber,
                    'refills_remaining': item.refills_remaining,
                    'refill_after_date': item.refill_after_date,
                    'insurance_covered': item.insurance_covered,
                    'insurance_paid': f"${item.insurance_paid:.2f}" if item.insurance_paid else None,
                    'copay': f"${item.copay:.2f}" if item.copay else None,
                    'insurance_savings': f"${item.insurance_savings:.2f}" if item.insurance_savings else None,
                    
                    # Electronics fields
                    'brand': item.brand,
                    'model_number': item.model_number,
                    'tech_specs': item.tech_specs,
                    'warranty_period': item.warranty_period,
                    'warranty_expiry': item.warranty_expiry,
                    'extended_warranty': item.extended_warranty,
                    'extended_warranty_period': item.extended_warranty_period,
                    'configuration': item.configuration,
                    'accessories': item.accessories,
                    
                    # Fashion fields
                    'size': item.size,
                    'color': item.color,
                    'style': item.style,
                    'material': item.material,
                    'care_instructions': item.care_instructions,
                    'fit': item.fit,
                    'original_price': f"${item.original_price:.2f}" if item.original_price else None,
                    'savings': f"${item.savings:.2f}" if item.savings else None,
                    'personalization': item.personalization,
                    
                    # Grocery fields
                    'organic': item.organic,
                    'locally_grown': item.locally_grown,
                    'expiration_date': item.expiration_date,
                    'weight_unit': item.weight_unit,
                    'unit_measure': item.unit_measure,
                    'price_per_unit': f"${item.price_per_unit:.2f}" if item.price_per_unit else None,
                    'price_per_weight': f"${item.price_per_weight:.2f}" if item.price_per_weight else None,
                    'on_sale': item.on_sale,
                    'substituted': item.substituted,
                    'original_item': item.original_item,
                    
                    # Home improvement fields
                    'dimensions': item.dimensions,
                    'finish': item.finish,
                    'color_hex': item.color_hex,
                    'coverage': item.coverage,
                    'capacity': item.capacity,
                    'measure_unit': item.measure_unit,
                    'price_per_measure': f"${item.price_per_measure:.4f}" if item.price_per_measure else None,
                    'installation_required': item.installation_required,
                    'warranty': item.warranty,
                    
                    # Wholesale fields
                    'units_per_case': item.units_per_case,
                    'case_quantity': item.case_quantity,
                    'case_qty': item.case_quantity,  # Alias
                    'case_unit': item.unit or 'case',  # Unit type for cases
                    'price_per_case': item.price_per_case,
                    'cases_per_pallet': item.cases_per_pallet,
                    'pallet_quantity': item.pallet_quantity,
                    'price_per_pallet': item.price_per_pallet,
                    'moq': item.moq,
                    'lead_time': item.lead_time,
                    'manufacturer': item.manufacturer,
                    'country_of_origin': item.country_of_origin,
                    
                    # Display format fields (for templates that show formatted prices)
                    'unit_price_display': f"${item.unit_price:.2f}",
                    'total_display': f"${item.total:.2f}",
                    'member_price': f"${item.unit_price * 0.9:.2f}" if random.random() < 0.3 else None,  # 10% member discount
                    'bulk_deal': random.choice(['3 for $5', 'Buy 2 Get 1', '10% off 6+', None]),
                    'promo_text': item.promotion if item.promotion else None,
                    
                    # Shopify inventory upload fields
                    'mpn': item.mpn,  # Manufacturer Part Number
                    'msrp': f"${item.msrp:.2f}" if item.msrp else None,  # Manufacturer Suggested Retail Price
                    'msrp_raw': item.msrp,  # For calculations
                    'map_price': f"${item.map_price:.2f}" if item.map_price else None,  # Minimum Advertised Price
                    'map_price_raw': item.map_price,  # For calculations
                    'case_pack': item.case_pack or item.units_per_case,  # Units per case (with fallback)
                    
                    # Additional wholesale fields
                    'volume_discount_percentage': item.volume_discount_percentage,
                    'volume_discount_amount': item.volume_discount_amount,
                    'tiered_pricing': item.tiered_pricing,
                    'pricing_tiers': item.pricing_tiers,
                    'price_tier': item.price_tier,
                    'show_tiered_pricing_table': item.show_tiered_pricing_table,
                    'stock_status': item.stock_status,
                    'available_quantity': item.available_quantity,
                    'line_total': item.line_total if item.line_total is not None else item.total,
                    'total_units': item.total_units,
                    'promotional_discount': item.promotional_discount,
                    'member_discount': item.member_discount,
                    'deposit': item.deposit,
                    'tiered_discount': item.tiered_discount,
                    'total_price': item.total_price,
                    
                    # Digital product fields
                    'product_type': item.product_type,
                    'platform': item.platform,
                    'version': item.version,
                    'license_type': item.license_type,
                    'license_seats': item.license_seats,
                    'subscription_period': item.subscription_period,
                    'renewal_date': item.renewal_date,
                    'auto_renew': item.auto_renew,
                    'download_url': item.download_url,
                    'access_url': item.access_url,
                    'activation_code': item.activation_code,
                    'license_key': item.license_key,
                    'file_size': item.file_size,
                    'system_requirements': item.system_requirements,
                    'expiry_date': item.expiry_date,
                    
                    # QSR fields
                    'modifiers': item.modifiers,
                    'special_instructions': item.special_instructions,
                    'is_combo': item.is_combo,
                    'combo_savings': f"${item.combo_savings:.2f}" if item.combo_savings else None,
                    
                    # Marketplace fields
                    'seller_name': item.seller_name,
                    'seller_sku': item.seller_sku,
                    'condition': item.condition,
                    'variant': item.variant,
                    
                    # Shopify product classification fields
                    'category': item.category,
                    'collection': item.collection,
                    
                    # Additional fields
                    'discount_description': item.discount_description,
                    'attributes': self._build_attributes_string(item),  # Built from specialized fields
                    'tax_indicator': '*' if item.tax_rate > 0 else None,  # For dense receipts
                }
                for item in receipt.line_items
            ],
            
            # Alias for template compatibility (some templates use 'items' instead of 'line_items')
            'items': [
                {
                    # Base fields
                    'description': item.description,
                    'name': item.name or item.description,
                    'quantity': item.quantity,
                    'unit_price': item.unit_price,
                    'total': item.total,
                    'total_price': f"${item.total:.2f}",  # Formatted alias for wholesale templates
                    'upc': item.upc,
                    'sku': item.sku,
                    'unit': item.unit,
                    'tax_rate': item.tax_rate,
                    'tax_amount': item.tax_amount,
                    'qty': item.quantity,
                    'rate': item.unit_price,
                    'price': item.unit_price,
                    'unit_cost': item.unit_price,
                    'amount': item.total,
                    'discount': item.discount if item.discount > 0 else None,
                    
                    # Wholesale fields
                    'units_per_case': item.units_per_case,
                    'case_quantity': item.case_quantity,
                    'price_per_case': item.price_per_case,
                    'moq': item.moq,
                    'lead_time': item.lead_time,
                    'manufacturer': item.manufacturer,
                    'country_of_origin': item.country_of_origin,
                    'volume_discount_amount': item.volume_discount_amount,
                    'stock_status': item.stock_status,
                    'total_units': item.total_units,
                    'brand': item.brand,
                    'line_total': item.line_total if item.line_total is not None else item.total,  # Alias for template compatibility
                    
                    # E-commerce fields
                    'variant': item.variant,
                    'personalization': item.personalization,
                    'size': item.size,
                    'color': item.color,
                    'material': item.material,
                    'category': item.category,
                    'collection': item.collection,
                    'style': item.style,
                    'model_number': item.model_number,
                    'mpn': item.mpn,
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
            'receipt_number': receipt.transaction_number,  # Alias for templates
            'order_number': receipt.transaction_number or receipt.invoice_number,  # Alias for QSR templates
            'transaction_time': receipt.transaction_time,
            'transaction_date': receipt.invoice_date or receipt.order_date,  # Alias for templates
            
            # Miscellaneous
            'terms_and_conditions': receipt.terms_and_conditions,
            'note': receipt.note,
            'notes': receipt.notes,
            'return_policy': receipt.return_policy,
            'return_days': 30,  # Standard return window (used by many templates)
            'footer_message': receipt.footer_message,
            
            # Wholesale B2B
            'po_number': receipt.po_number,
            'shipping_terms': receipt.shipping_terms,
            'account_manager': receipt.account_manager,
            'bank_details': receipt.bank_details,
            'tax_exempt': receipt.tax_exempt,
            
            # B2B Billing fields (for wholesale templates)
            'billing_company_name': receipt.buyer_name,
            'billing_contact_person': f"{self.fake.first_name()} {self.fake.last_name()}",
            'billing_street_address': receipt.buyer_address.split(',')[0] if receipt.buyer_address and ',' in receipt.buyer_address else receipt.buyer_address,
            'billing_city': receipt.buyer_address.split(',')[1].strip() if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 1 else '',
            'billing_state': receipt.buyer_address.split(',')[2].split()[0] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 else '',
            'billing_zip': receipt.buyer_address.split(',')[2].split()[1] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 and len(receipt.buyer_address.split(',')[2].split()) > 1 else '',
            'billing_email': receipt.buyer_email,
            'billing_phone': receipt.buyer_phone,
            
            # B2B Shipping company fields
            'shipping_company_name': receipt.buyer_name,
            'shipping_contact_person': f"{self.fake.first_name()} {self.fake.last_name()}",
            'shipping_email': receipt.buyer_email,
            'shipping_phone': receipt.buyer_phone,
            
            # B2B Account fields
            'account_status': random.choice(['Active', 'Premium', 'Preferred', 'Standard']),
            'account_tier': random.choice(['Gold', 'Silver', 'Bronze', 'Platinum', 'Enterprise']),
            'account_discount': f"${random.uniform(50, 500):.2f}",
            'account_discount_percentage': f"{random.randint(5, 20)}%",
            'account_manager_name': f"{self.fake.first_name()} {self.fake.last_name()}",
            'account_manager_email': f"sales@{self.fake.domain_name()}",
            'account_manager_phone': self.fake.phone_number(),
            'credit_line': f"${random.randint(10000, 100000):,}",
            'credit_available': f"${random.randint(5000, 50000):,}",
            'credit_used': f"${random.randint(1000, 20000):,}",
            'credit_status': random.choice(['Good Standing', 'Current', 'Approved']),
            'ytd_purchases': f"${random.randint(10000, 500000):,}",
            
            # B2B Company address (supplier side)
            'company_address': receipt.supplier_address,
            'company_city': receipt.supplier_address.split(',')[1].strip() if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 1 else '',
            'company_state': receipt.supplier_address.split(',')[2].split()[0] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 else '',
            'company_zip': receipt.supplier_address.split(',')[2].split()[1] if receipt.supplier_address and ',' in receipt.supplier_address and len(receipt.supplier_address.split(',')) > 2 and len(receipt.supplier_address.split(',')[2].split()) > 1 else '',
            
            # B2B Shipping logistics
            'ship_date': (datetime.now() + timedelta(days=random.randint(1, 3))).strftime('%B %d, %Y'),
            'delivery_method': random.choice(['Ground', 'Express', 'Freight', 'LTL', 'FTL']),
            'shipping_service': random.choice(['Standard Ground', 'Express Shipping', 'Freight Delivery']),
            'shipping_instructions': random.choice(['Dock delivery required', 'Liftgate needed', 'Call before delivery', '']),
            'fob_point': random.choice(['Origin', 'Destination', 'Shipping Point']),
            'freight_class': random.choice(['50', '70', '85', '100', '125']),
            'freight_cost': f"${random.uniform(50, 500):.2f}",
            'total_weight': f"{random.randint(50, 2000)} lbs",
            'pallet_count': random.randint(1, 10),
            'handling_fee': f"${random.uniform(10, 50):.2f}",
            'fuel_surcharge': f"${random.uniform(5, 30):.2f}",
            'liftgate_fee': f"${random.uniform(25, 75):.2f}",
            
            # B2B Payment terms
            'early_payment_discount': f"${random.uniform(50, 200):.2f}",
            'early_payment_discount_percentage': f"{random.randint(1, 3)}%",
            'early_payment_days': random.choice([10, 15]),
            'early_payment_terms': random.choice(['2/10 Net 30', '1/15 Net 30', '2/10 Net 45']),
            'late_fee_percentage': f"{random.uniform(1.5, 3):.1f}%",
            'restocking_fee': f"{random.randint(10, 25)}%",
            'volume_discount': f"${random.uniform(100, 1000):.2f}",
            'promotional_discount': f"${random.uniform(25, 150):.2f}",
            'subtotal_after_discount': f"${receipt.subtotal * 0.9:.2f}",
            
            # B2B Portal and tracking
            'portal_url': f"https://portal.{self.fake.domain_name()}",
            'tracking_url': f"https://track.{self.fake.domain_name()}/order/",
            'current_year': datetime.now().year,
            'customer_service_email': f"support@{self.fake.domain_name()}",
            
            # E-commerce
            'shipping_method': receipt.shipping_method,
            'delivery_estimate': receipt.delivery_estimate,
            'gift_message': receipt.gift_message,
            'gift_wrap_charge': f"${random.uniform(2.99, 7.99):.2f}" if random.random() < 0.2 else None,
            
            # Order/Payment status fields
            'order_status': random.choice(['Confirmed', 'Processing', 'Shipped', 'Delivered', 'Complete']),
            'payment_status': random.choice(['Paid', 'Completed', 'Authorized', 'Captured']),
            'payment_date': (datetime.now() - timedelta(days=random.randint(0, 5))).strftime('%B %d, %Y'),
            'payment_status_class': 'status-paid',  # CSS class
            'shipping_status_class': 'status-shipped',  # CSS class
            'delivery_status_class': 'status-delivered',  # CSS class
            'status_date': datetime.now().strftime('%B %d, %Y'),
            'status_icon': '✓',
            
            # Marketplace fields
            'buyer_username': f"buyer_{random.randint(1000, 9999)}",
            'marketplace_address': receipt.supplier_address,
            'marketplace_fee': f"${random.uniform(5, 50):.2f}",
            'marketplace_tagline': random.choice(['Shop with confidence', 'Trusted marketplace', 'Quality guaranteed']),
            'protection_days': random.choice([30, 45, 60, 90]),
            'seller_shipping_charge': f"${random.uniform(0, 15):.2f}",
            
            # Digital/Account fields
            'account_id': f"ACC-{random.randint(100000, 999999)}",
            'purchase_date': datetime.now().strftime('%B %d, %Y'),
            'refund_window': random.choice(['7 days', '14 days', '30 days']),
            
            # Bank details (for blue_wave_invoice)
            'bank_name': random.choice(['Chase Bank', 'Bank of America', 'Wells Fargo', 'Citibank', 'US Bank']),
            'bank_account': f"****{random.randint(1000, 9999)}",
            
            # Social media
            'social_media': f"@{self.fake.user_name()}",
            
            # Shipping address fields (for online orders)
            'shipping_name': receipt.buyer_name,
            'shipping_address': receipt.buyer_address,
            'shipping_street_address': receipt.buyer_address.split(',')[0] if receipt.buyer_address and ',' in receipt.buyer_address else receipt.buyer_address,
            'shipping_city': receipt.buyer_address.split(',')[1].strip() if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 1 else '',
            'shipping_state': receipt.buyer_address.split(',')[2].split()[0] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 else '',
            'shipping_zip': receipt.buyer_address.split(',')[2].split()[1] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 and len(receipt.buyer_address.split(',')[2].split()) > 1 else '',
            'shipping_charge': f"${receipt.shipping_cost:.2f}" if hasattr(receipt, 'shipping_cost') and receipt.shipping_cost > 0 else "$0.00",
            'carrier_name': random.choice(['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac', 'Amazon Logistics']),
            'expected_delivery': (datetime.now() + timedelta(days=random.randint(2, 7))).strftime('%B %d, %Y'),
            'estimated_delivery': (datetime.now() + timedelta(days=random.randint(2, 7))).strftime('%B %d, %Y'),
            'estimated_delivery_date': (datetime.now() + timedelta(days=random.randint(2, 7))).strftime('%Y-%m-%d'),
            
            # Delivery fields (for grocery/food delivery)
            'delivery_address': receipt.buyer_address,
            'delivery_city': receipt.buyer_address.split(',')[1].strip() if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 1 else '',
            'delivery_state': receipt.buyer_address.split(',')[2].split()[0] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 else '',
            'delivery_zip': receipt.buyer_address.split(',')[2].split()[1] if receipt.buyer_address and ',' in receipt.buyer_address and len(receipt.buyer_address.split(',')) > 2 and len(receipt.buyer_address.split(',')[2].split()) > 1 else '',
            'delivery_recipient': receipt.buyer_name,
            'delivery_fee': f"${random.uniform(2.99, 9.99):.2f}",
            'delivery_window': random.choice(['9am-12pm', '12pm-3pm', '3pm-6pm', '6pm-9pm', 'ASAP']),
            'delivery_status': random.choice(['Delivered', 'In Transit', 'Out for Delivery', 'Processing']),
            'delivery_instructions': random.choice(['Leave at door', 'Ring doorbell', 'Hand to customer', '', '']),
            
            # Support/Contact fields
            'support_email': f"support@{self.fake.domain_name()}",
            'support_phone': self.fake.phone_number(),
            'customer_service_phone': self.fake.phone_number(),
            'store_email': f"store@{self.fake.domain_name()}",
            'returns_url': f"https://returns.{self.fake.domain_name()}",
            'support_url': f"https://support.{self.fake.domain_name()}",
            'tech_support_phone': self.fake.phone_number(),
            
            # Return policy fields
            'return_window': 30,
            'return_deadline': (datetime.now() + timedelta(days=30)).strftime('%B %d, %Y'),
            'electronics_return_days': 15,
            
            # Loyalty
            'loyalty_points_earned': receipt.loyalty_points_earned,
            'loyalty_points_balance': receipt.loyalty_points_balance,
            'loyalty_rewards_available': receipt.loyalty_rewards_available,
            
            # Survey
            'survey_url': receipt.survey_url,
            'survey_code': receipt.survey_code,
            'survey_prize': random.choice(['$1,000 Gift Card', 'Free Item', '20% Off Next Purchase', '$500 Cash']),
            'survey_valid_days': random.choice([7, 14, 30]),
            'survey_reward': random.choice(['Free Coffee', 'Free Burger', '$5 Off', '10% Discount']),
            'survey_id': f"SRV{random.randint(100000, 999999)}",
            
            # Barcode
            'barcode_value': receipt.barcode_value,
            'barcode_image': receipt.barcode_image,
            
            # Marketplace seller info (order-level)
            'seller_name': receipt.seller_name,
            'seller_username': receipt.seller_username,
            'seller_rating': receipt.seller_rating,
            'seller_reviews': receipt.seller_reviews,
            'marketplace_name': receipt.marketplace_name
        }
        
        # Detect if this is a grocery order and organize items by temperature zone
        # Check if any line item has grocery fields (organic, expiration_date, weight_unit)
        has_grocery_items = any(
            hasattr(item, 'organic') or 
            hasattr(item, 'expiration_date') or 
            hasattr(item, 'weight_unit') or
            hasattr(item, 'locally_grown')
            for item in receipt.line_items
        ) if receipt.line_items else False
        
        if has_grocery_items:
            # Organize items by temperature zone for grocery template
            grocery_zones = self._organize_grocery_items_by_zone(receipt.line_items)
            base_dict.update(grocery_zones)
        else:
            # For non-grocery orders, use placeholder counts
            base_dict.update({
                'produce_items': random.randint(0, 10),
                'refrigerated_items': random.randint(0, 8),
                'frozen_items': random.randint(0, 5),
                'pantry_items': random.randint(0, 15),
            })
        
        return base_dict
    
    def generate_for_template(self,
                             template_name: str,
                             min_items: int = 2,
                             max_items: int = 5,
                             locale: Optional[str] = None) -> RetailReceiptData:
        """
        Generate invoice data with realistic product categories for a specific template
        
        This method automatically selects appropriate product categories based on the
        template type (e.g., Costco gets groceries, Amazon gets electronics mix, etc.)
        
        Args:
            template_name: Name of template (e.g., 'ebay_invoice', 'costco_invoice')
            min_items: Minimum number of items
            max_items: Maximum number of items
            locale: Optional locale override
            
        Returns:
            RetailReceiptData with mixed product categories appropriate for the template
            
        Example:
            >>> generator = RetailDataGenerator()
            >>> receipt = generator.generate_for_template('ebay_invoice', min_items=10, max_items=10)
            >>> # Will generate 35% electronics, 25% toys, 15% sports, 15% home, 10% books
        """
        try:
            from generators.template_category_mapper import get_category_mapper
            mapper = get_category_mapper()
            
            # Generate mixed categories based on template
            num_items = random.randint(min_items, max_items)
            categories = mapper.generate_mixed_categories(template_name, num_items)
            
            # Get primary category for brand name generation
            primary_category = mapper.get_primary_category(template_name)
            
            # Generate order with mixed categories
            return self.generate_online_order(
                store_type=primary_category,
                min_items=num_items,
                max_items=num_items,
                locale=locale,
                mixed_categories=categories
            )
        except ImportError:
            # Fallback if template_category_mapper not available
            return self.generate_online_order(
                store_type='fashion',
                min_items=min_items,
                max_items=max_items,
                locale=locale
            )


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
