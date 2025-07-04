#!/usr/bin/env python3
"""
Real-time tax dashboard for monitoring tax implications
"""
import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.tax_monitor import TaxMonitor


def load_positions():
    """Load current positions (mock data for demo)"""
    # In production, this would connect to the trading engine
    return {
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'quantity': 0.5,
            'entry_price': 45000,
            'current_price': 48000,
            'opened_at': datetime(2024, 1, 15),
            'side': 'buy'
        },
        'ETH/USDT': {
            'symbol': 'ETH/USDT',
            'quantity': 10,
            'entry_price': 2500,
            'current_price': 2300,
            'opened_at': datetime(2024, 6, 1),
            'side': 'buy'
        },
        'SOL/USDT': {
            'symbol': 'SOL/USDT',
            'quantity': 100,
            'entry_price': 100,
            'current_price': 120,
            'opened_at': datetime(2023, 12, 1),
            'side': 'buy'
        }
    }


def create_position_table(positions, tax_monitor):
    """Create positions table with tax info"""
    table = Table(title="📊 Current Positions - Tax Analysis", expand=True)
    
    table.add_column("Symbol", style="cyan")
    table.add_column("P&L", justify="right")
    table.add_column("Term", style="yellow")
    table.add_column("Days Held", justify="right")
    table.add_column("Tax Rate", justify="right")
    table.add_column("Tax Impact", justify="right", style="red")
    table.add_column("After Tax", justify="right", style="green")
    table.add_column("Recommendation", style="magenta")
    
    for symbol, pos in positions.items():
        # Create mock position object
        class Position:
            def __init__(self, data):
                self.symbol = data['symbol']
                self.quantity = data['quantity']
                self.entry_price = data['entry_price']
                self.current_price = data['current_price']
                self.opened_at = data['opened_at']
                self.side = data['side']
        
        position = Position(pos)
        analysis = tax_monitor.analyze_position_tax_impact(position, pos['current_price'])
        
        # Format P&L with color
        pnl_text = f"${analysis['potential_pnl']:,.2f}"
        pnl_color = "green" if analysis['potential_pnl'] > 0 else "red"
        
        # Format tax impact
        tax_text = f"${analysis['tax_liability']:,.2f}"
        after_tax_text = f"${analysis['after_tax_profit']:,.2f}"
        
        table.add_row(
            symbol,
            Text(pnl_text, style=pnl_color),
            analysis['term_type'],
            str(analysis['holding_days']),
            f"{analysis['tax_rate']*100:.0f}%",
            tax_text,
            after_tax_text,
            analysis['recommendation'][:30] + "..."
        )
    
    return table


def create_summary_panel(summary):
    """Create summary panel"""
    content = f"""
[bold cyan]Unrealized Gains/Losses:[/bold cyan]
  Short-term: ${summary['unrealized_gains']['short_term']:,.2f}
  Long-term:  ${summary['unrealized_gains']['long_term']:,.2f}
  [bold]Total:      ${summary['unrealized_gains']['total']:,.2f}[/bold]

[bold yellow]Tax Estimates:[/bold yellow]
  Estimated Tax: ${summary['estimated_tax']:,.2f}
  Effective Rate: {summary['effective_rate']*100:.1f}%
  
[bold green]YTD Realized:[/bold green]
  Gains:  ${summary['ytd_realized_gains']:,.2f}
  Losses: ${summary['ytd_realized_losses']:,.2f}
  
[bold magenta]Quarterly Payment:[/bold magenta]
  Next Due: ${summary['quarterly_estimate']:,.2f}
"""
    return Panel(content, title="💰 Tax Summary", expand=True)


def create_opportunities_table(opportunities, title):
    """Create opportunities table"""
    table = Table(title=title, expand=True)
    
    table.add_column("Symbol", style="cyan")
    table.add_column("Action", style="yellow")
    table.add_column("Tax Benefit", justify="right", style="green")
    table.add_column("Details")
    
    for opp in opportunities:
        if 'tax_benefit' in opp and opp['tax_benefit'] > 0:
            benefit = f"${opp['tax_benefit']:,.2f}"
            action = "Harvest Loss"
        elif 'potential_tax_savings' in opp:
            benefit = f"${opp['potential_tax_savings']:,.2f}"
            action = f"Wait {opp['days_to_long_term']}d"
        else:
            benefit = "-"
            action = "Review"
        
        details = f"P&L: ${opp['potential_pnl']:,.2f}"
        
        table.add_row(
            opp['symbol'],
            action,
            benefit,
            details
        )
    
    return table


def create_tax_dashboard():
    """Create the main dashboard layout"""
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    tax_monitor = TaxMonitor(config['tax'])
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=20),
        Layout(name="opportunities", size=10),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="positions", ratio=2),
        Layout(name="summary", ratio=1)
    )
    
    layout["opportunities"].split_row(
        Layout(name="harvest"),
        Layout(name="wait")
    )
    
    # Header
    layout["header"].update(
        Panel(
            "[bold blue]🧮 Real-Time Tax Monitor[/bold blue]\n"
            f"Jurisdiction: {config['tax']['jurisdiction']} | "
            f"Method: {config['tax']['method']} | "
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            expand=True
        )
    )
    
    # Load positions and calculate
    positions = load_positions()
    current_prices = {k: v['current_price'] for k, v in positions.items()}
    summary = tax_monitor.get_portfolio_tax_summary(positions, current_prices)
    
    # Update panels
    layout["positions"].update(create_position_table(positions, tax_monitor))
    layout["summary"].update(create_summary_panel(summary))
    
    # Opportunities
    if summary['harvest_opportunities']:
        layout["harvest"].update(
            create_opportunities_table(
                summary['harvest_opportunities'][:3],
                "🎯 Tax Loss Harvesting"
            )
        )
    else:
        layout["harvest"].update(Panel("No loss harvesting opportunities", title="🎯 Tax Loss Harvesting"))
    
    if summary['wait_opportunities']:
        layout["wait"].update(
            create_opportunities_table(
                summary['wait_opportunities'][:3],
                "⏳ Wait for Long-Term"
            )
        )
    else:
        layout["wait"].update(Panel("No positions near long-term threshold", title="⏳ Wait for Long-Term"))
    
    # Footer
    layout["footer"].update(
        Panel(
            "[dim]Press Ctrl+C to exit | Updates every 5 seconds | "
            "Run 'python scripts/generate_tax_report.py' for full report[/dim]",
            expand=True
        )
    )
    
    return layout


def main():
    """Run the tax dashboard"""
    console = Console()
    
    console.print("\n[bold]Starting Tax Dashboard...[/bold]\n")
    
    try:
        with Live(create_tax_dashboard(), refresh_per_second=0.2) as live:
            while True:
                time.sleep(5)
                live.update(create_tax_dashboard())
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    # Check if rich is installed
    try:
        import rich
    except ImportError:
        print("Please install rich: pip install rich")
        sys.exit(1)
    
    main()