def format_price(price: float) -> str:
    """Format price with euro symbol and thousands separator"""
    return f"€{price:,.0f}"