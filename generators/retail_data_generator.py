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


class RetailDataGenerator:
    """Generates retail-specific synthetic data ensuring ALL 37 entities appear"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        self.fake = Faker(locale)
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # Retail product categories
        self.grocery_items = [
            "Organic Bananas", "Milk 2%", "Whole Wheat Bread", "Free Range Eggs",
            "Greek Yogurt", "Avocados", "Cherry Tomatoes", "Baby Spinach",
            "Orange Juice", "Cheddar Cheese", "Ground Beef", "Chicken Breast"
        ]
        
        self.electronics_items = [
            "USB-C Cable", "Wireless Mouse", "Bluetooth Headphones", "Phone Case",
            "Screen Protector", "Portable Charger", "HDMI Cable", "Keyboard",
            "Webcam", "Memory Card 64GB", "Laptop Stand", "Cable Organizer"
        ]
        
        self.clothing_items = [
            "Cotton T-Shirt", "Denim Jeans", "Running Shoes", "Hoodie",
            "Baseball Cap", "Socks Pack", "Polo Shirt", "Shorts",
            "Sneakers", "Jacket", "Dress Shirt", "Belt"
        ]
        
        self.pharmacy_items = [
            "Ibuprofen 200mg", "Vitamin D3", "Band-Aids", "Hand Sanitizer",
            "Face Masks", "Multivitamin", "Pain Relief Cream", "Cough Syrup",
            "Allergy Medicine", "First Aid Kit", "Thermometer", "Lip Balm"
        ]
        
        self.fuel_items = [
            "Unleaded Regular", "Unleaded Premium", "Diesel", "Car Wash Basic",
            "Windshield Fluid", "Motor Oil 5W-30", "Air Freshener", "Energy Drink"
        ]
        
        # Payment methods (PAYMENT_METHOD)
        self.payment_methods = [
            "Visa", "Mastercard", "American Express", "Discover",
            "Debit Card", "Cash", "Gift Card", "Apple Pay", "Google Pay"
        ]
        
        # Store types
        self.store_types = {
            'grocery': ['Fresh Market', 'Save-A-Lot', 'QuickMart', 'Daily Grocers'],
            'electronics': ['Tech World', 'Gadget Hub', 'Electronics Plus', 'Digital Store'],
            'clothing': ['Fashion Boutique', 'Style Shop', 'Trendy Threads', 'Apparel Co'],
            'pharmacy': ['HealthCare Pharmacy', 'MedPlus', 'WellRx', 'Community Pharmacy'],
            'fuel': ['QuickFuel', 'Gas & Go', 'FuelMart', 'Express Station'],
            'qsr': ["Joe's Diner", 'Burger Palace', 'Pizza Corner', 'Taco Express'],
            'retail': ['MegaMart', 'SuperStore', 'Value Shop', 'Discount Depot']
        }
    
    def generate_line_item(self, category: str = 'grocery') -> RetailLineItem:
        """Generate a single line item with complete entity coverage"""
        
        # Select product based on category
        if category == 'grocery':
            description = random.choice(self.grocery_items)
            unit = random.choice(['ea', 'lb', 'oz', 'gal', 'pkg'])
        elif category == 'electronics':
            description = random.choice(self.electronics_items)
            unit = 'ea'
        elif category == 'clothing':
            description = random.choice(self.clothing_items)
            unit = 'ea'
        elif category == 'pharmacy':
            description = random.choice(self.pharmacy_items)
            unit = 'ea'
        elif category == 'fuel':
            description = random.choice(self.fuel_items)
            unit = 'gal'
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
        if category == 'pharmacy' and random.random() < 0.5:
            lot_number = self.fake.bothify(text='LOT##??####')
        if category == 'electronics' and random.random() < 0.3:
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
                            store_type: str = 'grocery',
                            min_items: int = 3,
                            max_items: int = 8) -> RetailReceiptData:
        """Generate a complete POS receipt with ALL 37 entities"""
        
        receipt = RetailReceiptData()
        
        # Document metadata
        receipt.doc_type = "Receipt"
        receipt.invoice_number = self.fake.bothify(text='REC-######')
        receipt.invoice_date = self.fake.date_this_year().strftime('%m/%d/%Y')
        # ORDER_DATE typically not on POS receipts
        
        # Merchant information
        receipt.supplier_name = random.choice(self.store_types.get(store_type, self.store_types['retail']))
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
                             store_type: str = 'electronics',
                             min_items: int = 2,
                             max_items: int = 6) -> RetailReceiptData:
        """Generate an online order/invoice with ALL 37 entities"""
        
        receipt = self.generate_pos_receipt(store_type=store_type, min_items=min_items, max_items=max_items)
        
        # Override for online orders
        receipt.doc_type = "Invoice"
        receipt.invoice_number = self.fake.bothify(text='ORD-######')
        
        # ORDER_DATE (for online orders)
        order_date = self.fake.date_between(start_date='-30d', end_date='today')
        receipt.order_date = order_date.strftime('%m/%d/%Y')
        receipt.invoice_date = (order_date + timedelta(days=random.randint(0, 2))).strftime('%m/%d/%Y')
        
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
    # Test generation
    generator = RetailDataGenerator(seed=42)
    
    print("=== POS Receipt ===")
    pos_receipt = generator.generate_pos_receipt(store_type='grocery', min_items=5, max_items=8)
    print(f"Receipt: {pos_receipt.invoice_number}")
    print(f"Store: {pos_receipt.supplier_name}")
    print(f"Register: {pos_receipt.register_number}, Cashier: {pos_receipt.cashier_id}")
    print(f"Items: {len(pos_receipt.line_items)}")
    print(f"Subtotal: ${pos_receipt.subtotal:.2f}")
    print(f"Tax ({pos_receipt.tax_rate}%): ${pos_receipt.tax_amount:.2f}")
    print(f"Total: ${pos_receipt.total_amount:.2f}")
    print(f"Payment: {pos_receipt.payment_method} - {pos_receipt.payment_terms}")
    
    print("\n=== Online Order ===")
    online_order = generator.generate_online_order(store_type='electronics', min_items=3, max_items=5)
    print(f"Order: {online_order.invoice_number}")
    print(f"Customer: {online_order.buyer_name}")
    print(f"Tracking: {online_order.tracking_number}")
    print(f"Total: ${online_order.total_amount:.2f}")
