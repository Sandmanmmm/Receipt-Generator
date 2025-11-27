"""
Synthetic Data Generator
Generates randomized invoice data for templates
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import random
from faker import Faker


@dataclass
class InvoiceItem:
    """Represents a single line item on an invoice"""
    description: str
    quantity: int
    rate: float
    tax_rate: float = 0.0
    
    @property
    def amount(self) -> float:
        """Calculate item total amount"""
        return self.quantity * self.rate


@dataclass
class InvoiceData:
    """Complete invoice data structure"""
    # Company info
    company_name: str
    company_address: str
    company_phone: str
    company_email: str
    company_tax_id: Optional[str] = None
    
    # Invoice info
    invoice_number: str
    invoice_date: str
    due_date: Optional[str] = None
    
    # Client info
    client_name: str
    client_address: str
    client_phone: Optional[str] = None
    client_email: Optional[str] = None
    
    # Shipping info
    ship_to: Optional[Dict[str, str]] = None
    
    # Items
    items: List[InvoiceItem] = field(default_factory=list)
    
    # Pricing
    currency_symbol: str = "$"
    subtotal: float = 0.0
    discount: float = 0.0
    discount_percent: float = 0.0
    tax: float = 0.0
    tax_rate: float = 0.0
    shipping: float = 0.0
    total: float = 0.0
    
    # Additional fields
    include_tax: bool = False
    payment_info: Optional[Dict[str, str]] = None
    notes: Optional[str] = None
    terms: Optional[str] = None
    
    def calculate_totals(self):
        """Calculate all invoice totals"""
        self.subtotal = sum(item.amount for item in self.items)
        
        if self.discount_percent > 0:
            self.discount = self.subtotal * (self.discount_percent / 100)
        
        taxable_amount = self.subtotal - self.discount
        
        if self.tax_rate > 0:
            self.tax = taxable_amount * (self.tax_rate / 100)
        
        self.total = taxable_amount + self.tax + self.shipping


class SyntheticDataGenerator:
    """Generates synthetic invoice data with randomization"""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        """
        Initialize the generator
        
        Args:
            locale: Locale for faker (e.g., 'en_US', 'en_GB', 'de_DE')
            seed: Random seed for reproducibility
        """
        self.fake = Faker(locale)
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # Product/service descriptions
        self.products = [
            "Web Development Services", "Graphic Design", "Content Writing",
            "SEO Optimization", "Social Media Management", "Consulting Hours",
            "Software License", "Cloud Hosting", "Domain Registration",
            "Email Marketing Campaign", "Logo Design", "Website Maintenance",
            "Mobile App Development", "Database Management", "UI/UX Design",
            "Video Production", "Photography Services", "Copywriting",
            "Translation Services", "Technical Support", "Training Session",
            "Project Management", "Data Analysis", "Network Setup",
            "Security Audit", "Code Review", "API Integration"
        ]
        
        # Payment methods
        self.payment_methods = [
            "Bank Transfer", "Credit Card", "PayPal", "Check",
            "Wire Transfer", "ACH", "Stripe", "Square"
        ]
        
        # Currency symbols
        self.currencies = {
            'en_US': '$', 'en_GB': '£', 'de_DE': '€',
            'fr_FR': '€', 'ja_JP': '¥', 'en_CA': 'C$'
        }
    
    def generate_company_data(self) -> Dict[str, Any]:
        """Generate random company information"""
        return {
            'company_name': self.fake.company(),
            'company_address': self.fake.address().replace('\n', ', '),
            'company_phone': self.fake.phone_number(),
            'company_email': self.fake.company_email(),
            'company_tax_id': self.fake.bothify(text='##-#######') if random.random() > 0.3 else None
        }
    
    def generate_client_data(self, include_shipping: bool = False) -> Dict[str, Any]:
        """Generate random client information"""
        data = {
            'client_name': self.fake.name() if random.random() > 0.5 else self.fake.company(),
            'client_address': self.fake.address().replace('\n', ', '),
            'client_phone': self.fake.phone_number() if random.random() > 0.6 else None,
            'client_email': self.fake.email() if random.random() > 0.5 else None
        }
        
        if include_shipping and random.random() > 0.7:
            data['ship_to'] = {
                'name': self.fake.name(),
                'address': self.fake.address().replace('\n', ', ')
            }
        
        return data
    
    def generate_invoice_info(self, days_back: int = 90) -> Dict[str, Any]:
        """Generate invoice number and dates"""
        invoice_date = self.fake.date_between(
            start_date=f'-{days_back}d',
            end_date='today'
        )
        
        include_due_date = random.random() > 0.3
        due_date = None
        if include_due_date:
            days_until_due = random.choice([7, 14, 30, 45, 60])
            due_date = invoice_date + timedelta(days=days_until_due)
        
        return {
            'invoice_number': self.fake.bothify(text='INV-####-????').upper(),
            'invoice_date': invoice_date.strftime('%Y-%m-%d'),
            'due_date': due_date.strftime('%Y-%m-%d') if due_date else None
        }
    
    def generate_items(self, min_items: int = 1, max_items: int = 10) -> List[InvoiceItem]:
        """Generate random invoice line items"""
        num_items = random.randint(min_items, max_items)
        items = []
        
        for _ in range(num_items):
            description = random.choice(self.products)
            quantity = random.randint(1, 20)
            rate = round(random.uniform(10, 1000), 2)
            tax_rate = random.choice([0, 5, 7.5, 10, 15, 20]) if random.random() > 0.5 else 0
            
            items.append(InvoiceItem(
                description=description,
                quantity=quantity,
                rate=rate,
                tax_rate=tax_rate
            ))
        
        return items
    
    def generate_payment_info(self) -> Optional[Dict[str, str]]:
        """Generate payment information"""
        if random.random() < 0.4:
            return None
        
        method = random.choice(self.payment_methods)
        info = {'method': method}
        
        if method in ['Bank Transfer', 'Wire Transfer', 'ACH']:
            info['account'] = f"Account: {self.fake.bothify(text='####-####-####')}"
            info['instructions'] = "Please include invoice number in payment reference"
        elif method in ['PayPal', 'Stripe']:
            info['account'] = f"Email: {self.fake.email()}"
        
        return info
    
    def generate_notes_and_terms(self) -> Dict[str, Optional[str]]:
        """Generate notes and terms"""
        notes_options = [
            None,
            "Thank you for your business. Please pay within the specified due date.",
            "All work has been completed as per the agreed specifications.",
            "Please contact us if you have any questions about this invoice.",
            "We appreciate your continued partnership."
        ]
        
        terms_options = [
            None,
            "Payment is due within 30 days of invoice date. Late payments may incur a 1.5% monthly fee.",
            "All prices are in USD. Goods remain the property of the seller until paid in full.",
            "Please make checks payable to the company name listed above.",
            "Disputes must be raised within 7 days of invoice date."
        ]
        
        return {
            'notes': random.choice(notes_options),
            'terms': random.choice(terms_options)
        }
    
    def generate_invoice(self,
                        min_items: int = 1,
                        max_items: int = 10,
                        include_shipping: bool = True,
                        include_discount: bool = True,
                        include_tax: bool = True) -> InvoiceData:
        """
        Generate a complete random invoice
        
        Args:
            min_items: Minimum number of line items
            max_items: Maximum number of line items
            include_shipping: Whether to add shipping charges
            include_discount: Whether to apply discounts
            include_tax: Whether to include tax
            
        Returns:
            InvoiceData object with all fields populated
        """
        # Generate all components
        company_data = self.generate_company_data()
        client_data = self.generate_client_data(include_shipping)
        invoice_info = self.generate_invoice_info()
        items = self.generate_items(min_items, max_items)
        payment_info = self.generate_payment_info()
        notes_terms = self.generate_notes_and_terms()
        
        # Create invoice
        invoice = InvoiceData(
            **company_data,
            **client_data,
            **invoice_info,
            items=items,
            currency_symbol=self.currencies.get(self.fake.locale, '$'),
            include_tax=include_tax,
            payment_info=payment_info,
            **notes_terms
        )
        
        # Add optional charges
        if include_discount and random.random() > 0.6:
            invoice.discount_percent = random.choice([5, 10, 15, 20])
        
        if include_tax and random.random() > 0.3:
            invoice.tax_rate = random.choice([5, 7.5, 10, 15, 20])
        
        if include_shipping and random.random() > 0.5:
            invoice.shipping = round(random.uniform(5, 50), 2)
        
        # Calculate totals
        invoice.calculate_totals()
        
        return invoice
    
    def invoice_to_dict(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Convert InvoiceData to dictionary for template rendering"""
        return {
            'company_name': invoice.company_name,
            'company_address': invoice.company_address,
            'company_phone': invoice.company_phone,
            'company_email': invoice.company_email,
            'company_tax_id': invoice.company_tax_id,
            'invoice_number': invoice.invoice_number,
            'invoice_date': invoice.invoice_date,
            'due_date': invoice.due_date,
            'client_name': invoice.client_name,
            'client_address': invoice.client_address,
            'client_phone': invoice.client_phone,
            'client_email': invoice.client_email,
            'ship_to': invoice.ship_to,
            'items': [
                {
                    'description': item.description,
                    'quantity': item.quantity,
                    'rate': item.rate,
                    'tax_rate': item.tax_rate,
                    'amount': item.amount
                }
                for item in invoice.items
            ],
            'currency_symbol': invoice.currency_symbol,
            'subtotal': invoice.subtotal,
            'discount': invoice.discount,
            'discount_percent': invoice.discount_percent,
            'tax': invoice.tax,
            'tax_rate': invoice.tax_rate,
            'shipping': invoice.shipping,
            'total': invoice.total,
            'include_tax': invoice.include_tax,
            'payment_info': invoice.payment_info,
            'notes': invoice.notes,
            'terms': invoice.terms
        }


if __name__ == '__main__':
    # Example usage
    generator = SyntheticDataGenerator(locale='en_US', seed=42)
    invoice = generator.generate_invoice(min_items=3, max_items=8)
    
    print(f"Invoice {invoice.invoice_number}")
    print(f"Company: {invoice.company_name}")
    print(f"Client: {invoice.client_name}")
    print(f"Items: {len(invoice.items)}")
    print(f"Total: {invoice.currency_symbol}{invoice.total:.2f}")
