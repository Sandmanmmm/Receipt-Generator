"""
Modern Invoice Generator
Generates enhanced invoice data for modern professional templates
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random
from faker import Faker
from .data_generator import SyntheticDataGenerator, InvoiceData, InvoiceItem
from .production_randomizer import ProductionRandomizer
try:
    from .visual_assets import VisualAssetGenerator
except ImportError:
    VisualAssetGenerator = None

@dataclass
class ModernInvoiceItem(InvoiceItem):
    """Enhanced invoice item with SKU"""
    sku: Optional[str] = None
    unit_price: float = 0.0  # Alias for rate to match template
    total: float = 0.0       # Alias for amount to match template

    def __post_init__(self):
        # Ensure aliases are set
        if self.unit_price == 0.0 and self.rate != 0.0:
            self.unit_price = self.rate
        if self.total == 0.0:
            self.total = self.quantity * self.unit_price

@dataclass
class ModernInvoiceData(InvoiceData):
    """Enhanced invoice data for modern templates"""
    # Additional fields for modern templates
    po_number: Optional[str] = None
    brand_primary_color: str = "#2c3e50"
    
    # Bank details
    bank_name: Optional[str] = None
    account_number: Optional[str] = None
    routing_number: Optional[str] = None
    payment_method: Optional[str] = None
    
    # Override items to use ModernInvoiceItem
    items: List[Any] = field(default_factory=list)
    
    # Template compatibility aliases
    tax_rate_str: str = ""  # Formatted tax rate (e.g. "8%")

class ModernInvoiceGenerator(SyntheticDataGenerator):
    """Generates data specifically for modern invoice templates"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        super().__init__(locale, seed)
        self.locale = locale
        
        # Brand colors for modern templates
        self.brand_colors = [
            "#2c3e50", "#3498db", "#2ecc71", "#e74c3c", 
            "#9b59b6", "#f39c12", "#1abc9c", "#34495e",
            "#16a085", "#27ae60", "#2980b9", "#8e44ad",
            "#d35400", "#c0392b", "#7f8c8d"
        ]
        
        # Initialize visual asset generator
        self.visual_gen = VisualAssetGenerator() if VisualAssetGenerator else None
        
    def generate_modern_invoice(self, 
                               min_items: int = 3, 
                               max_items: int = 8) -> Dict[str, Any]:
        """
        Generate a complete dictionary of data for modern invoice templates
        
        Returns:
            Dictionary compatible with Jinja2 templates
        """
        # Generate base data using parent class methods
        company = self.generate_company_data()
        client = self.generate_client_data()
        invoice_info = self.generate_invoice_info()
        
        # Generate enhanced items
        items = []
        num_items = random.randint(min_items, max_items)
        
        subtotal = 0.0
        for _ in range(num_items):
            description = random.choice(self.products)
            quantity = random.randint(1, 10)
            price = round(random.uniform(50, 2000), 2)
            amount = quantity * price
            subtotal += amount
            
            items.append({
                'sku': f"SKU-{self.fake.bothify(text='####')}",
                'description': description,
                'quantity': quantity,
                'unit_price': price,
                'amount': amount,
                'tax_rate': 0.0,  # Simplified for now
                'attributes': None  # Optional product attributes for ecommerce templates
            })
            
        # Calculate totals
        tax_rate = 0.08
        tax = subtotal * tax_rate
        total = subtotal + tax
        
        # Generate banking info
        bank_name = self.fake.company() + " Bank"
        account_num = self.fake.bothify(text='****####')
        routing_num = self.fake.bothify(text='****####')
        
        # Generate visual assets
        logo = None
        qr_code = None
        if self.visual_gen:
            # Generate logo with primary brand color
            # Convert hex to RGB for logo generator
            hex_color = random.choice(self.brand_colors)
            r = int(hex_color.lstrip('#')[0:2], 16)
            g = int(hex_color.lstrip('#')[2:4], 16)
            b = int(hex_color.lstrip('#')[4:6], 16)
            rgb_color = (r, g, b)
            logo = self.visual_gen.generate_logo(company['company_name'], color=rgb_color)
            
            # Generate QR code for invoice number
            qr_code = self.visual_gen.generate_qr_code(invoice_info['invoice_number'])
        else:
            hex_color = random.choice(self.brand_colors)

        # Combine all data
        data = {
            # Company
            'company_name': company['company_name'],
            'company_address': company['company_address'],
            'company_phone': company['company_phone'],
            'company_email': company['company_email'],
            'company_city': self.fake.city() + ", " + self.fake.state_abbr() + " " + self.fake.zipcode(),
            'company_website': self.fake.url(),
            'logo': logo,
            
            # Client
            'client_name': client['client_name'],
            'client_address': client['client_address'],
            'client_contact': client.get('client_email', ''),
            'client_city': self.fake.city() + ", " + self.fake.state_abbr() + " " + self.fake.zipcode(),
            'ship_to': client.get('ship_to', None),
            
            # Invoice Details
            'invoice_number': invoice_info['invoice_number'],
            'invoice_date': invoice_info['invoice_date'],
            'due_date': invoice_info['due_date'],
            'po_number': f"PO-{self.fake.bothify(text='#####')}",
            'qr_code': qr_code,
            
            # Items & Totals
            'items': items,
            'line_items': items,  # Alias for SimplePNGRenderer compatibility
            'subtotal': subtotal,
            'tax_rate': f"{tax_rate*100:.0f}%",
            'tax': tax,
            'total': total,
            'currency_symbol': self.currencies.get(self.locale, '$'),
            
            # Payment & Terms - use B2B context
            'payment_terms': random.choice(["Net 30", "Due on Receipt", "Net 15", "Net 60"]),
            'payment_method': ProductionRandomizer.get_payment_method(context='b2b')[0],
            'bank_name': bank_name,
            'account_number': account_num,
            'routing_number': routing_num,
            
            # Branding
            'brand_primary_color': hex_color,
            
            # Text
            'notes': random.choice([
                "Thank you for your business!",
                "Please include invoice number on your check.",
                "We appreciate your prompt payment."
            ]),
            'terms': "Payment is due within 30 days. Late payments may incur interest."
        }
        
        return data

    def to_dict(self, invoice: Any) -> Dict[str, Any]:
        """Passthrough helper to mirror other generators.
        
        Adds template compatibility aliases for field names that differ
        between generators and templates (e.g., supplier_name vs company_name).
        """
        if isinstance(invoice, dict):
            data = invoice
        elif hasattr(invoice, "__dataclass_fields__"):
            from dataclasses import asdict
            data = asdict(invoice)
        else:
            data = dict(invoice)
        
        # Add template compatibility aliases
        # Amazon/ecommerce templates use supplier_name, we have company_name
        if 'company_name' in data and 'supplier_name' not in data:
            data['supplier_name'] = data['company_name']
        if 'company_address' in data and 'supplier_address' not in data:
            data['supplier_address'] = data['company_address']
        
        # Some templates use _amount suffix for financial fields
        if 'subtotal' in data and 'subtotal_amount' not in data:
            data['subtotal_amount'] = data['subtotal']
        if 'tax' in data and 'tax_amount' not in data:
            data['tax_amount'] = data['tax']
        if 'total' in data and 'total_amount' not in data:
            data['total_amount'] = data['total']
        if 'discount' in data and 'discount_amount' not in data:
            data['discount_amount'] = data['discount']
        if 'shipping' in data and 'shipping_cost' not in data:
            data['shipping_cost'] = data['shipping']
        
        # Fix item field naming: templates expect unit_price and total, dataclass uses rate
        for items_key in ['items', 'line_items']:
            if items_key in data and isinstance(data[items_key], list):
                for item in data[items_key]:
                    if isinstance(item, dict):
                        # Add unit_price alias for rate
                        if 'rate' in item and 'unit_price' not in item:
                            item['unit_price'] = item['rate']
                        if 'unit_price' in item and 'price' not in item:
                            item['price'] = item['unit_price']
                        
                        # Calculate total/amount if missing
                        if 'total' not in item and 'amount' not in item:
                            qty = item.get('quantity', 1)
                            unit_price = item.get('unit_price', item.get('rate', item.get('price', 0)))
                            item['total'] = round(qty * unit_price, 2)
                            item['amount'] = item['total']
                        elif 'total' not in item and 'amount' in item:
                            item['total'] = item['amount']
                        elif 'amount' not in item and 'total' in item:
                            item['amount'] = item['total']
        
        # Ensure line_items alias exists
        if 'items' in data and 'line_items' not in data:
            data['line_items'] = data['items']
        
        return data
