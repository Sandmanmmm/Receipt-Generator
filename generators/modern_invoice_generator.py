"""
Modern Invoice Generator
Generates enhanced invoice data for modern professional templates
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random
from faker import Faker
from .data_generator import SyntheticDataGenerator, InvoiceData, InvoiceItem
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
                'tax_rate': 0.0  # Simplified for now
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
            
            # Payment & Terms
            'payment_terms': random.choice(["Net 30", "Due on Receipt", "Net 15"]),
            'payment_method': random.choice(["Bank Transfer", "Credit Card", "Check"]),
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
