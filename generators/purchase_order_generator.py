"""
Purchase Order Generator
Generates B2B/wholesale purchase order data for PO templates
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random
from datetime import datetime, timedelta
from faker import Faker
from .data_generator import SyntheticDataGenerator
from .production_randomizer import ProductionRandomizer


@dataclass
class PurchaseOrderItem:
    """Purchase order line item"""
    line_number: int
    sku: str
    part_number: str
    description: str
    product_name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    quantity: int = 1
    uom: str = "EA"  # Unit of measure: EA, CS, PK, BX, etc.
    unit_price: float = 0.0
    unit_cost: float = 0.0  # Alias for unit_price
    total: float = 0.0
    amount: float = 0.0  # Alias for total
    # Optional fields
    upc: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    weight: Optional[str] = None
    origin_country: Optional[str] = None
    hs_code: Optional[str] = None  # Harmonized System code for international trade
    buyer_sku: Optional[str] = None  # For dropship SKU mapping
    
    def __post_init__(self):
        # Ensure aliases are set
        if self.unit_cost == 0.0 and self.unit_price != 0.0:
            self.unit_cost = self.unit_price
        elif self.unit_price == 0.0 and self.unit_cost != 0.0:
            self.unit_price = self.unit_cost
        
        if self.total == 0.0:
            self.total = self.quantity * self.unit_price
        if self.amount == 0.0:
            self.amount = self.total


@dataclass
class PurchaseOrderData:
    """Complete purchase order data structure"""
    po_number: str
    po_date: str
    supplier: Dict[str, str]
    buyer: Dict[str, str]
    ship_to: Dict[str, str]
    payment_terms: str
    shipping_method: str
    line_items: List[Dict[str, Any]]
    subtotal: float
    tax: float
    tax_rate: float
    tax_amount: float
    shipping: float
    shipping_cost: float
    total: float
    total_amount: float
    # All optional fields from generation
    status: Optional[str] = None
    requested_delivery_date: Optional[str] = None
    delivery_date: Optional[str] = None
    ship_date: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_company: Optional[str] = None
    supplier_address: Optional[str] = None
    supplier_city: Optional[str] = None
    supplier_state: Optional[str] = None
    supplier_zip: Optional[str] = None
    supplier_phone: Optional[str] = None
    supplier_email: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_company: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_city: Optional[str] = None
    buyer_state: Optional[str] = None
    buyer_zip: Optional[str] = None
    buyer_phone: Optional[str] = None
    buyer_email: Optional[str] = None
    ship_to_name: Optional[str] = None
    ship_to_company: Optional[str] = None
    ship_to_address: Optional[str] = None
    ship_to_city: Optional[str] = None
    ship_to_state: Optional[str] = None
    ship_to_zip: Optional[str] = None
    ship_to_phone: Optional[str] = None
    tax_rate_percent: float = 0.0
    discount: float = 0.0
    discount_percent: float = 0.0
    grand_total: float = 0.0
    notes: Optional[str] = None
    terms: Optional[str] = None
    terms_conditions: Optional[str] = None
    special_instructions: str = ""
    authorized_by: Optional[str] = None
    brand_primary_color: Optional[str] = None
    brand_accent_color: Optional[str] = None
    currency_symbol: str = "$"
    currency: str = "USD"
    # Type-specific fields
    incoterms: Optional[str] = None
    port_of_loading: Optional[str] = None
    port_of_discharge: Optional[str] = None
    port_loading: Optional[str] = None
    port_discharge: Optional[str] = None
    etd_eta: Optional[str] = None
    customs_broker: Optional[str] = None
    letter_of_credit: Optional[str] = None
    container_type: Optional[str] = None
    inspection_required: bool = False
    blind_ship: bool = False
    custom_packaging: Optional[str] = None
    inserts_marketing: Optional[str] = None
    packing_slip_header: Optional[str] = None
    return_address: Optional[str] = None


class PurchaseOrderGenerator(SyntheticDataGenerator):
    """Generates purchase order data for B2B/wholesale transactions"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        super().__init__(locale, seed)
        self.locale = locale
        self.fake = Faker(locale)
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # PO-specific data
        self.statuses = ['PENDING', 'APPROVED', 'SHIPPED', 'RECEIVED', 'CANCELLED']
        self.payment_terms_b2b = ['Net 30', 'Net 60', 'Net 90', '2/10 Net 30', '1/15 Net 45', 
                                   'COD', 'Prepayment', 'Letter of Credit', 'Upon Receipt']
        self.shipping_methods = ['Ground', 'Air Freight', 'Ocean Freight', 'LTL', 'FTL', 
                                'Express', 'Standard', 'Two-Day', 'Overnight']
        self.incoterms = ['FOB', 'CIF', 'EXW', 'DDP', 'DAP', 'FCA', 'CFR']
        self.ports = {
            'Asia': ['Shanghai, China', 'Shenzhen, China', 'Hong Kong', 'Singapore', 'Busan, South Korea'],
            'USA': ['Los Angeles, USA', 'New York, USA', 'Seattle, USA', 'Houston, USA', 'Miami, USA'],
            'Europe': ['Rotterdam, Netherlands', 'Hamburg, Germany', 'Antwerp, Belgium']
        }
        self.currencies = ['USD', 'EUR', 'GBP', 'CNY', 'JPY']
        
        # Units of measure
        self.uoms = ['EA', 'CS', 'PK', 'BX', 'PC', 'UN', 'CT', 'DZ', 'PR', 'ST']
        
        # Product categories for wholesale
        self.wholesale_categories = [
            'Electronics', 'Apparel', 'Home Goods', 'Office Supplies', 
            'Industrial Tools', 'Beauty Products', 'Food & Beverage',
            'Toys & Games', 'Sporting Goods', 'Automotive Parts'
        ]
        
        # Phase 7B: Industry-specific product catalogs with specialized fields
        self.industry_products = {
            'beauty': [
                # Makeup
                {'name': 'Foundation SPF 15', 'shade': 'Natural Beige', 'shade_hex': '#E8B793'},
                {'name': 'Liquid Lipstick', 'shade': 'Ruby Red', 'shade_hex': '#C41E3A'},
                {'name': 'Mascara Waterproof', 'shade': 'Midnight Black', 'shade_hex': '#000000'},
                {'name': 'Eyeshadow Palette', 'shade': 'Rose Gold', 'shade_hex': '#B76E79'},
                {'name': 'Blush Powder', 'shade': 'Peachy Pink', 'shade_hex': '#FFB3BA'},
                {'name': 'Concealer Stick', 'shade': 'Fair Ivory', 'shade_hex': '#F5E6D3'},
                {'name': 'Lip Gloss', 'shade': 'Clear Shimmer', 'shade_hex': '#FFC0CB'},
                # Skincare
                {'name': 'Vitamin C Serum', 'ingredients': 'Ascorbic Acid, Hyaluronic Acid, Vitamin E'},
                {'name': 'Retinol Night Cream', 'ingredients': 'Retinol 0.5%, Peptides, Ceramides'},
                {'name': 'Hydrating Face Mask', 'ingredients': 'Aloe Vera, Glycerin, Rose Water'},
                {'name': 'Exfoliating Scrub', 'ingredients': 'Jojoba Beads, Vitamin E, Tea Tree Oil'},
                {'name': 'Anti-Aging Moisturizer', 'ingredients': 'Hyaluronic Acid, Collagen, SPF 30'},
                # Hair Care
                {'name': 'Keratin Shampoo', 'ingredients': 'Keratin Protein, Argan Oil, Biotin'},
                {'name': 'Color-Safe Conditioner', 'ingredients': 'UV Filters, Panthenol, Coconut Oil'},
                {'name': 'Hair Serum', 'ingredients': 'Argan Oil, Vitamin E, Silicone'},
            ],
            
            'electronics': [
                {'name': 'Wireless Bluetooth Earbuds', 'specs': ['Bluetooth 5.0', '8hr battery', 'IPX4 water resistant'], 'warranty': '1 Year'},
                {'name': 'USB-C Fast Charger 65W', 'specs': ['65W PD', 'GaN technology', 'Foldable plug'], 'warranty': '2 Years'},
                {'name': 'Portable Power Bank 20000mAh', 'specs': ['20000mAh capacity', 'Dual USB-C ports', 'Quick Charge 3.0'], 'warranty': '1 Year'},
                {'name': 'LED Desk Lamp', 'specs': ['15W LED', 'Touch control', '5 brightness levels'], 'warranty': '1 Year'},
                {'name': 'Wireless Mouse 2.4GHz', 'specs': ['1600 DPI', '2.4GHz wireless', '12-month battery'], 'warranty': '1 Year'},
                {'name': 'HDMI Cable 6ft', 'specs': ['HDMI 2.0', '4K@60Hz', 'Gold-plated connectors'], 'warranty': 'Lifetime'},
                {'name': 'USB Hub 7-Port', 'specs': ['USB 3.0', '7 ports', '5Gbps transfer speed'], 'warranty': '18 Months'},
                {'name': 'Webcam 1080P HD', 'specs': ['1080P resolution', 'Auto-focus', 'Built-in microphone'], 'warranty': '2 Years'},
                {'name': 'Bluetooth Speaker Portable', 'specs': ['20W output', '12hr battery', 'Waterproof IPX7'], 'warranty': '1 Year'},
                {'name': 'Laptop Stand Aluminum', 'specs': ['Adjustable height', 'Aluminum alloy', 'Supports up to 15.6"'], 'warranty': '1 Year'},
                {'name': 'Mechanical Keyboard RGB', 'specs': ['Cherry MX switches', 'RGB backlight', 'N-key rollover'], 'warranty': '2 Years'},
            ],
            
            'fashion': [
                {'name': 'Cotton T-Shirt', 'style_code': 'TS-001', 'colors': ['Black', 'White', 'Navy', 'Gray', 'Red'], 'sizes': ['S', 'M', 'L', 'XL', 'XXL']},
                {'name': 'Slim Fit Jeans', 'style_code': 'JN-205', 'colors': ['Dark Blue', 'Light Blue', 'Black'], 'sizes': ['28', '30', '32', '34', '36', '38']},
                {'name': 'Hoodie Pullover', 'style_code': 'HD-430', 'colors': ['Black', 'Gray', 'Navy', 'Maroon'], 'sizes': ['S', 'M', 'L', 'XL']},
                {'name': 'Athletic Leggings', 'style_code': 'LG-112', 'colors': ['Black', 'Navy', 'Charcoal'], 'sizes': ['XS', 'S', 'M', 'L', 'XL']},
                {'name': 'Flannel Shirt', 'style_code': 'SH-678', 'colors': ['Red Plaid', 'Blue Plaid', 'Green Plaid'], 'sizes': ['S', 'M', 'L', 'XL']},
                {'name': 'Cargo Pants', 'style_code': 'CP-334', 'colors': ['Khaki', 'Olive', 'Black'], 'sizes': ['30', '32', '34', '36', '38']},
                {'name': 'V-Neck Sweater', 'style_code': 'SW-892', 'colors': ['Burgundy', 'Navy', 'Gray'], 'sizes': ['S', 'M', 'L', 'XL']},
                {'name': 'Denim Jacket', 'style_code': 'DJ-445', 'colors': ['Blue', 'Black', 'Light Wash'], 'sizes': ['S', 'M', 'L', 'XL']},
                {'name': 'Polo Shirt', 'style_code': 'PS-223', 'colors': ['White', 'Black', 'Navy', 'Red'], 'sizes': ['S', 'M', 'L', 'XL', 'XXL']},
            ],
            
            'food_beverage': [
                {'name': 'Organic Coffee Beans', 'storage': 'Ambient', 'storage_temp': 'Room Temp', 'shelf_life': '12 months'},
                {'name': 'Greek Yogurt 32oz', 'storage': 'Refrigerated', 'storage_temp': '35-40°F', 'shelf_life': '30 days'},
                {'name': 'Frozen Pizza 12"', 'storage': 'Frozen', 'storage_temp': '0°F or below', 'shelf_life': '12 months'},
                {'name': 'Granola Bars Box/24', 'storage': 'Ambient', 'storage_temp': 'Room Temp', 'shelf_life': '6 months'},
                {'name': 'Fresh Orange Juice 64oz', 'storage': 'Refrigerated', 'storage_temp': '35-40°F', 'shelf_life': '14 days'},
                {'name': 'Protein Powder 2lb', 'storage': 'Ambient', 'storage_temp': 'Room Temp', 'shelf_life': '24 months'},
                {'name': 'Almond Milk Half Gallon', 'storage': 'Refrigerated', 'storage_temp': '35-40°F', 'shelf_life': '7 days'},
                {'name': 'Ice Cream Pint', 'storage': 'Frozen', 'storage_temp': '0°F or below', 'shelf_life': '6 months'},
                {'name': 'Pasta Sauce 24oz', 'storage': 'Ambient', 'storage_temp': 'Room Temp', 'shelf_life': '18 months'},
                {'name': 'Energy Drinks 12-Pack', 'storage': 'Ambient', 'storage_temp': 'Room Temp', 'shelf_life': '12 months'},
            ],
            
            'home_goods': [
                {'name': 'Office Chair Ergonomic', 'dimensions': '25"W x 25"D x 40"H', 'assembly': True, 'material': 'Mesh/Steel'},
                {'name': 'LED Floor Lamp', 'dimensions': '9"W x 9"D x 65"H', 'assembly': False, 'material': 'Metal/Plastic'},
                {'name': 'Storage Ottoman', 'dimensions': '18"W x 18"D x 18"H', 'assembly': False, 'material': 'Fabric/Wood'},
                {'name': 'Bookshelf 5-Tier', 'dimensions': '24"W x 12"D x 72"H', 'assembly': True, 'material': 'Wood Composite'},
                {'name': 'Area Rug 5x7', 'dimensions': '60"W x 84"L', 'assembly': False, 'material': 'Polypropylene'},
                {'name': 'Coffee Table Modern', 'dimensions': '48"W x 24"D x 18"H', 'assembly': True, 'material': 'Glass/Metal'},
                {'name': 'Desk Organizer Set', 'dimensions': '12"W x 8"D x 5"H', 'assembly': False, 'material': 'Bamboo'},
                {'name': 'Curtains Blackout 84"', 'dimensions': '52"W x 84"L', 'assembly': False, 'material': 'Polyester'},
                {'name': 'Wall Mirror 24x36', 'dimensions': '24"W x 36"H', 'assembly': False, 'material': 'Glass/MDF'},
            ],
            
            'manufacturing': [
                {'name': 'Industrial LED Panel 100W', 'moq': 100, 'lead_time': '45 days', 'specs': '100W, IP65, 5000K'},
                {'name': 'Aluminum Extrusion 6063-T5', 'moq': 500, 'lead_time': '30 days', 'specs': '6063-T5, Custom length'},
                {'name': 'Injection Molded Case', 'moq': 1000, 'lead_time': '60 days', 'specs': 'ABS plastic, Custom color'},
                {'name': 'PCB Assembly 2-Layer', 'moq': 250, 'lead_time': '21 days', 'specs': '2-layer, Lead-free solder'},
                {'name': 'CNC Machined Part', 'moq': 500, 'lead_time': '35 days', 'specs': 'Aluminum 6061, ±0.05mm tolerance'},
                {'name': 'Rubber Gasket Custom', 'moq': 1000, 'lead_time': '28 days', 'specs': 'EPDM rubber, Shore A 70'},
                {'name': 'Steel Bracket Stamped', 'moq': 2000, 'lead_time': '40 days', 'specs': 'Cold-rolled steel, Zinc plated'},
            ],
            
            'paper': [
                {'name': 'Thermal Receipt Paper 80mm', 'grade': 'Standard', 'gsm': 55, 'finish': 'Thermal Coated', 'brightness': '82'},
                {'name': 'Copy Paper A4 500 Sheets', 'grade': 'Premium', 'gsm': 80, 'finish': 'Matte', 'brightness': '96'},
                {'name': 'Cardstock 110lb White', 'grade': 'Cover', 'gsm': 300, 'finish': 'Smooth', 'brightness': '92'},
                {'name': 'Label Stock Roll 4x6', 'grade': 'Adhesive', 'gsm': 60, 'finish': 'Glossy', 'brightness': '88'},
                {'name': 'Kraft Paper Roll 36"', 'grade': 'Natural', 'gsm': 70, 'finish': 'Uncoated', 'brightness': '65'},
                {'name': 'Tissue Paper 20x30', 'grade': 'MG', 'gsm': 17, 'finish': 'Smooth', 'brightness': '90'},
                {'name': 'Parchment Paper Roll', 'grade': 'Silicone Coated', 'gsm': 41, 'finish': 'Non-stick', 'brightness': '85'},
            ]
        }
        
    def _generate_po_number(self) -> str:
        """Generate realistic purchase order number"""
        formats = [
            f"PO-{self.fake.year()}-{self.fake.bothify(text='#####')}",
            f"PO{self.fake.bothify(text='########')}",
            f"P{self.fake.random_number(digits=8)}",
            f"{self.fake.random_letter().upper()}{self.fake.random_letter().upper()}-{self.fake.bothify(text='####-####')}"
        ]
        return random.choice(formats)
    
    def _generate_po_item(self, line_num: int, po_type: str = 'domestic', industry: str = None) -> Dict[str, Any]:
        """Generate a single purchase order line item with optional industry-specific fields
        
        Args:
            line_num: Line item number
            po_type: PO type ('domestic', 'alibaba', 'dropship')
            industry: Optional industry for specialized fields
                     ('beauty', 'electronics', 'fashion', 'food_beverage', 
                      'home_goods', 'manufacturing', 'paper')
        """
        # Category-specific brands
        brand_map = {
            'Electronics': ['Samsung', 'LG', 'Sony', 'Panasonic', 'Dell', 'HP', 'Lenovo'],
            'Apparel': ['Nike', 'Adidas', 'Levi\'s', 'Gap', 'H&M', 'Zara', 'Uniqlo'],
            'Home Goods': ['Ikea', 'Target', 'Bed Bath', 'HomeGoods', 'Crate & Barrel'],
            'Office Supplies': ['Staples', '3M', 'Avery', 'Sharpie', 'Papermate', 'HP'],
            'Industrial Tools': ['DeWalt', 'Milwaukee', 'Ryobi', 'Black & Decker', 'Craftsman'],
            'Beauty Products': ['L\'Oreal', 'Maybelline', 'Revlon', 'CoverGirl', 'Neutrogena'],
            'Food & Beverage': ['Kraft', 'Nestle', 'Pepsi', 'Coca-Cola', 'General Mills'],
            'Toys & Games': ['Mattel', 'Hasbro', 'LEGO', 'Fisher-Price', 'Barbie'],
            'Sporting Goods': ['Nike', 'Adidas', 'Under Armour', 'Reebok', 'Puma'],
            'Automotive Parts': ['Bosch', 'ACDelco', 'Denso', 'Motorcraft', 'Fram']
        }
        
        # Determine category and product based on industry
        if industry and industry in self.industry_products:
            # Use industry-specific product catalog
            product_data = random.choice(self.industry_products[industry])
            category = industry.replace('_', ' ').title()
            product_name = product_data['name']
            
            # Map industry to brand category
            category_key = {
                'beauty': 'Beauty Products',
                'electronics': 'Electronics',
                'fashion': 'Apparel',
                'food_beverage': 'Food & Beverage',
                'home_goods': 'Home Goods',
                'manufacturing': 'Industrial Tools',
                'paper': 'Office Supplies'
            }.get(industry, 'Industrial Tools')
            
            brand = random.choice(brand_map.get(category_key, ['Generic Brand']))
        else:
            # Fallback to generic product selection
            category = random.choice(self.wholesale_categories)
            product_name = random.choice(self.products)
            brand = random.choice(brand_map.get(category, ['Generic Brand']))
            product_data = {}
        
        # Base item fields
        quantity = random.randint(10, 500)  # Wholesale quantities
        unit_price = round(random.uniform(5, 500), 2)
        total = quantity * unit_price
        
        item = {
            'line_number': line_num,
            'sku': f"{brand[:3].upper()}-{self.fake.bothify(text='####')}",
            'part_number': f"PN-{self.fake.bothify(text='######')}",
            'description': f"{brand} {product_name}",
            'product_name': product_name,
            'brand': brand,
            'category': category,
            'quantity': quantity,
            'uom': random.choice(self.uoms),
            'unit_price': unit_price,
            'unit_cost': unit_price,
            'total': total,
            'line_total': total,  # Alias for template compatibility
            'amount': total,
            'upc': self.fake.bothify(text='##########'),
            'model': f"{brand[:2].upper()}-{self.fake.bothify(text='###')}",
        }
        
        # Add industry-specific fields from product_data
        if industry == 'beauty':
            if 'shade' in product_data:
                item['shade'] = product_data['shade']
                item['shade_hex'] = product_data.get('shade_hex', '#000000')
            if 'ingredients' in product_data:
                item['ingredients'] = product_data['ingredients']
            item['batch_number'] = f"BT{self.fake.bothify(text='######')}"
        
        elif industry == 'electronics':
            if 'specs' in product_data:
                item['technical_specs'] = product_data['specs']
            if 'warranty' in product_data:
                item['warranty'] = product_data['warranty']
            item['serial_number'] = f"SN{self.fake.bothify(text='##########')}"
        
        elif industry == 'fashion':
            if 'style_code' in product_data:
                item['style_code'] = product_data['style_code']
                item['supplier_sku'] = product_data['style_code']  # Alias
            if 'colors' in product_data:
                item['color_name'] = random.choice(product_data['colors'])
                item['color_code'] = f"CLR-{self.fake.bothify(text='###')}"
            if 'sizes' in product_data:
                sizes = product_data['sizes']
                # Create size_run as dict with random quantities per size
                item['size_run'] = {size: random.randint(5, 50) for size in sizes}
                item['size'] = random.choice(sizes)
            # Fashion template uses quantity_ordered instead of quantity
            item['quantity_ordered'] = item['quantity']
        
        elif industry == 'food_beverage':
            item['lot_number'] = f"LOT{self.fake.bothify(text='#######')}"
            item['expiry_date'] = (datetime.now() + timedelta(days=random.randint(30, 730))).strftime('%Y-%m-%d')
            if 'storage_temp' in product_data:
                item['storage_temp'] = product_data['storage_temp']
            if 'storage' in product_data:
                item['storage_type'] = product_data['storage']
        
        elif industry == 'home_goods':
            if 'dimensions' in product_data:
                item['dimensions'] = product_data['dimensions']
            if 'assembly' in product_data:
                item['assembly_required'] = product_data['assembly']
            if 'material' in product_data:
                item['material'] = product_data['material']
        
        elif industry == 'manufacturing':
            if 'moq' in product_data:
                item['moq'] = product_data['moq']
            if 'lead_time' in product_data:
                item['lead_time'] = product_data['lead_time']
            if 'specs' in product_data:
                item['production_specs'] = product_data['specs']
        
        elif industry == 'paper':
            if 'grade' in product_data:
                item['paper_grade'] = product_data['grade']
            if 'gsm' in product_data:
                item['gsm'] = product_data['gsm']
            if 'finish' in product_data:
                item['finish'] = product_data['finish']
            if 'brightness' in product_data:
                item['brightness'] = product_data['brightness']
        
        # Add international trade fields for alibaba/dropship
        if po_type in ['alibaba', 'dropship']:
            item['origin_country'] = random.choice(['China', 'Taiwan', 'Vietnam', 'India'])
            item['hs_code'] = self.fake.bothify(text='####.##.##')
            item['weight'] = f"{random.randint(1, 50)} lbs"
        
        # Add dropship-specific fields
        if po_type == 'dropship':
            item['buyer_sku'] = f"DS-{self.fake.bothify(text='#####')}"
        
        return item
    
    def generate_purchase_order(self,
                               po_type: str = 'domestic',
                               min_items: int = 3,
                               max_items: int = 15,
                               layout: str = 'portrait',
                               industry: str = None) -> Dict[str, Any]:
        """
        Generate complete purchase order data
        
        Args:
            po_type: 'domestic', 'alibaba', 'dropship', 'generic', or 'landscape'
            min_items: Minimum number of line items
            max_items: Maximum number of line items
            layout: 'portrait' or 'landscape' (for layout-specific templates)
            industry: Optional industry for specialized templates
                     ('beauty', 'electronics', 'fashion', 'food_beverage', 
                      'home_goods', 'manufacturing', 'paper')
            
        Returns:
            Dictionary with all PO data for templates
        
        Note: Phase 7A added 'generic' and 'landscape' po_type support.
              Phase 7B added 'industry' parameter for industry-specific fields.
        """
        # Generate parties with proper address components
        def get_company_with_address():
            return {
                'name': self.fake.company(),
                'address': self.fake.street_address(),
                'city': self.fake.city(),
                'state': self.fake.state_abbr(),
                'zip': self.fake.zipcode(),
                'phone': self.fake.phone_number(),
                'email': self.fake.company_email()
            }
        
        supplier = get_company_with_address()
        buyer = get_company_with_address()
        ship_to = get_company_with_address()
        
        # Generate PO metadata
        po_number = self._generate_po_number()
        po_date = self.fake.date_between(start_date='-30d', end_date='today')
        requested_delivery = po_date + timedelta(days=random.randint(14, 90))
        status = random.choice(self.statuses)
        
        # Generate line items
        # Phase 7A: Map generic/landscape types to domestic for data generation
        # Phase 7B: Pass industry parameter for industry-specific fields
        data_po_type = po_type if po_type in ['domestic', 'alibaba', 'dropship'] else 'domestic'
        num_items = random.randint(min_items, max_items)
        line_items = [self._generate_po_item(i + 1, data_po_type, industry) for i in range(num_items)]
        
        # Calculate totals
        subtotal = sum(item['total'] for item in line_items)
        discount_percent = random.choice([0, 5, 10, 15])
        discount = subtotal * (discount_percent / 100) if discount_percent > 0 else 0
        tax_rate_percent = 0  # Wholesale POs often don't include tax
        tax = 0
        shipping_cost = round(random.uniform(50, 500), 2) if random.random() > 0.3 else 0
        total = subtotal - discount + tax + shipping_cost
        
        # Base PO data
        po_data = {
            # PO Information
            'po_number': po_number,
            'po_date': po_date.strftime('%Y-%m-%d'),
            'status': status,
            'requested_delivery_date': requested_delivery.strftime('%Y-%m-%d'),
            'ship_date': (po_date + timedelta(days=7)).strftime('%Y-%m-%d'),
            
            # Supplier (Vendor/Manufacturer)
            'supplier_name': supplier['name'],
            'supplier_company': supplier['name'],
            'supplier_address': supplier['address'],
            'supplier_city': supplier['city'],
            'supplier_state': supplier['state'],
            'supplier_zip': supplier['zip'],
            'supplier_phone': supplier['phone'],
            'supplier_email': supplier['email'],
            'supplier_account_number': f"ACCT-{self.fake.bothify(text='#####')}",
            
            # Buyer (Ordering Company)
            'buyer_company': buyer['name'],
            'buyer_address': buyer['address'],
            'buyer_city': buyer['city'],
            'buyer_state': buyer['state'],
            'buyer_zip': buyer['zip'],
            'buyer_phone': buyer['phone'],
            'buyer_email': buyer['email'],
            
            # Ship To Address
            'ship_to_name': ship_to['name'],
            'ship_to_company': ship_to['name'],
            'ship_to_address': ship_to['address'],
            'ship_to_city': ship_to['city'],
            'ship_to_state': ship_to['state'],
            'ship_to_zip': ship_to['zip'],
            'ship_to_phone': ship_to['phone'],
            
            # Terms
            'payment_terms': random.choice(self.payment_terms_b2b),
            'shipping_method': random.choice(self.shipping_methods),
            'fob': random.choice(['Origin', 'Destination', 'Shipping Point']),
            'fob_point': random.choice(['Origin', 'Destination']),
            
            # Line Items
            'line_items': line_items,
            'items': line_items,  # Alias
            
            # Totals
            'subtotal': subtotal,
            'discount': discount,
            'discount_amount': discount,  # Alias for template compatibility
            'discount_percent': discount_percent,
            'tax': tax,
            'tax_rate': tax_rate_percent,  # Numeric value
            'tax_rate_percent': tax_rate_percent,  # Numeric value
            'tax_amount': tax,  # Alias
            'shipping': shipping_cost,
            'shipping_cost': shipping_cost,
            'total': total,
            'total_amount': total,  # Alias
            'grand_total': total,
            'currency_symbol': '$',
            'currency': 'USD',
            
            # Notes
            'notes': random.choice([
                "Please confirm receipt of this PO within 24 hours.",
                "All items must meet quality specifications outlined in our vendor agreement.",
                "Partial shipments are acceptable with prior approval.",
                "Contact us immediately if any items are unavailable."
            ]),
            'terms': "Standard terms and conditions apply. See vendor agreement for details.",
            'special_instructions': ""
        }
        
        # Add type-specific fields
        if po_type == 'alibaba':
            po_data.update({
                'incoterms': random.choice([f"{inc} Shanghai" for inc in self.incoterms]),
                'port_of_loading': random.choice(self.ports['Asia']),
                'port_of_discharge': random.choice(self.ports['USA']),
                'etd_eta': f"{(po_date + timedelta(days=30)).strftime('%Y-%m-%d')} / {(po_date + timedelta(days=60)).strftime('%Y-%m-%d')}",
                'customs_broker': self.fake.company() + " Customs Brokers",
                'letter_of_credit': f"LC-{self.fake.bothify(text='########')}",
                'container_type': random.choice(['20ft', '40ft', '40ft HC']),
                'inspection_required': random.choice([True, False])
            })
        
        elif po_type == 'dropship':
            po_data.update({
                'blind_ship': True,
                'custom_packaging': random.choice([
                    'Plain packaging - no logos',
                    'Buyer branded packaging',
                    'White label packaging'
                ]),
                'inserts_marketing': random.choice([
                    'Include buyer catalog',
                    'No marketing materials',
                    'Include promotional flyer'
                ]),
                'packing_slip_header': f"{buyer['name']} - Order Confirmation",
                'return_address': f"{buyer['name']}, {buyer['address']}, {buyer['city']}, {buyer['state']} {buyer['zip']}"
            })
        
        return po_data
    
    def to_dict(self, purchase_order: dict) -> dict:
        """
        Convert purchase order data to dictionary (passthrough for compatibility)
        
        Args:
            purchase_order: Purchase order dict or PurchaseOrderData instance
            
        Returns:
            dict: Dictionary representation suitable for template rendering
        """
        # If already a dict, return as-is
        if isinstance(purchase_order, dict):
            return purchase_order
        
        # If it's a dataclass instance, convert to dict
        if hasattr(purchase_order, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(purchase_order)
        
        # Fallback: assume it's dict-like
        return dict(purchase_order)
        
        return po_data
