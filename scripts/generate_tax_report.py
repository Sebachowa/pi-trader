#!/usr/bin/env python3
"""
Generate tax reports for cryptocurrency trading
"""
import json
import argparse
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.tax_calculator import TaxCalculator


def load_trade_history(filename: str = 'data/trade_history.json'):
    """Load trade history from file"""
    if not os.path.exists(filename):
        print(f"❌ Trade history not found at {filename}")
        return []
    
    with open(filename, 'r') as f:
        return json.load(f)


def generate_tax_reports(year: int = None, output_dir: str = 'reports/taxes'):
    """Generate comprehensive tax reports"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    tax_config = config.get('tax', {})
    if year:
        tax_config['tax_year'] = year
    
    # Initialize calculator
    calculator = TaxCalculator(tax_config)
    
    # Load trade history
    trades = load_trade_history()
    
    if not trades:
        print("❌ No trades found. Make sure the bot has executed some trades.")
        return
    
    print(f"📊 Processing {len(trades)} trades...")
    
    # Add all transactions to calculator
    for trade in trades:
        # Skip if not in the tax year
        if 'timestamp' in trade:
            trade_year = datetime.fromisoformat(trade['timestamp']).year
            if year and trade_year != year:
                continue
        
        calculator.add_transaction(trade)
    
    # Generate summary
    summary = calculator.calculate_tax_summary()
    
    # Print summary to console
    print("\n" + "="*60)
    print(f"📋 TAX SUMMARY FOR {summary['tax_year']}")
    print("="*60)
    print(f"Method: {summary['method']}")
    print(f"Total Trades: {summary['summary']['number_of_trades']}")
    print()
    
    print("💰 GAINS/LOSSES:")
    print(f"  Short-term gains:  ${summary['summary']['short_term_gains']:,.2f}")
    print(f"  Short-term losses: ${summary['summary']['short_term_losses']:,.2f}")
    print(f"  Net short-term:    ${summary['summary']['net_short_term']:,.2f}")
    print()
    print(f"  Long-term gains:   ${summary['summary']['long_term_gains']:,.2f}")
    print(f"  Long-term losses:  ${summary['summary']['long_term_losses']:,.2f}")
    print(f"  Net long-term:     ${summary['summary']['net_long_term']:,.2f}")
    print()
    
    print("💸 TAX LIABILITY:")
    print(f"  Short-term tax:    ${summary['tax_liability']['short_term_tax']:,.2f}")
    print(f"  Long-term tax:     ${summary['tax_liability']['long_term_tax']:,.2f}")
    print(f"  TOTAL TAX DUE:     ${summary['tax_liability']['total_tax']:,.2f}")
    print(f"  Effective rate:    {summary['tax_liability']['effective_rate']*100:.1f}%")
    print()
    
    # Quarterly estimates
    if tax_config.get('quarterly_estimates'):
        print("📅 QUARTERLY ESTIMATES:")
        for quarter, data in summary['quarterly_estimates'].items():
            status = "✓ PAID" if data['paid'] else "⏳ DUE"
            print(f"  {quarter} (due {data['due_date']}): ${data['amount']:,.2f} {status}")
        print()
    
    # Export reports
    print("📁 GENERATING REPORTS...")
    
    # 1. IRS Form 8949
    form_8949_file = os.path.join(output_dir, f"form_8949_{summary['tax_year']}.csv")
    calculator.export_form_8949(form_8949_file)
    print(f"  ✓ Form 8949: {form_8949_file}")
    
    # 2. Summary JSON
    summary_file = os.path.join(output_dir, f"tax_summary_{summary['tax_year']}.json")
    calculator.export_summary_report(summary_file)
    print(f"  ✓ Summary: {summary_file}")
    
    # 3. Tax loss harvesting suggestions
    suggestions = calculator.generate_tax_loss_harvesting_suggestions()
    if suggestions:
        print("\n🎯 TAX LOSS HARVESTING OPPORTUNITIES:")
        for i, suggestion in enumerate(suggestions[:5]):  # Top 5
            print(f"  {i+1}. Sell {suggestion['quantity']:.4f} {suggestion['symbol']}")
            print(f"     Loss: ${suggestion['potential_loss']:,.2f}")
            print(f"     Tax savings: ${suggestion['tax_savings']:,.2f}")
    
    print("\n✅ Tax reports generated successfully!")
    
    # Additional export formats
    if 'turbotax' in tax_config.get('export_format', []):
        generate_turbotax_export(calculator, output_dir)
    
    if 'cointracker' in tax_config.get('export_format', []):
        generate_cointracker_export(calculator, output_dir)


def generate_turbotax_export(calculator: TaxCalculator, output_dir: str):
    """Generate TurboTax-compatible export"""
    filename = os.path.join(output_dir, f"turbotax_{calculator.tax_year}.csv")
    
    with open(filename, 'w') as f:
        f.write("Date Sold,Date Acquired,Symbol,Quantity,Cost Basis,Proceeds\n")
        
        for event in calculator.taxable_events:
            if event.timestamp.year == calculator.tax_year:
                buy_date = (event.timestamp - timedelta(days=event.holding_period_days))
                f.write(f"{event.timestamp.strftime('%m/%d/%Y')},")
                f.write(f"{buy_date.strftime('%m/%d/%Y')},")
                f.write(f"{event.symbol},{event.quantity:.8f},")
                f.write(f"{event.cost_basis:.2f},{event.proceeds:.2f}\n")
    
    print(f"  ✓ TurboTax export: {filename}")


def generate_cointracker_export(calculator: TaxCalculator, output_dir: str):
    """Generate CoinTracker-compatible export"""
    filename = os.path.join(output_dir, f"cointracker_{calculator.tax_year}.csv")
    
    with open(filename, 'w') as f:
        f.write("Date,Received Quantity,Received Currency,Sent Quantity,")
        f.write("Sent Currency,Fee Amount,Fee Currency,Tag\n")
        
        for event in calculator.taxable_events:
            if event.timestamp.year == calculator.tax_year:
                # Format for CoinTracker
                base, quote = event.symbol.split('/')
                f.write(f"{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')},")
                
                if event.side == 'sell':
                    f.write(f"{event.proceeds:.2f},{quote},")
                    f.write(f"{event.quantity:.8f},{base},")
                else:
                    f.write(f"{event.quantity:.8f},{base},")
                    f.write(f"{event.cost_basis:.2f},{quote},")
                
                f.write(f"{event.fee:.4f},{event.fee_currency},")
                f.write("trade\n")
    
    print(f"  ✓ CoinTracker export: {filename}")


def estimate_next_quarter_payment():
    """Estimate next quarterly tax payment"""
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    calculator = TaxCalculator(config['tax'])
    trades = load_trade_history()
    
    # Process only current year trades
    current_year = datetime.now().year
    ytd_trades = [t for t in trades 
                  if 'timestamp' in t and 
                  datetime.fromisoformat(t['timestamp']).year == current_year]
    
    for trade in ytd_trades:
        calculator.add_transaction(trade)
    
    summary = calculator.calculate_tax_summary()
    
    # Estimate based on YTD performance
    days_passed = datetime.now().timetuple().tm_yday
    days_in_year = 365
    projection_factor = days_in_year / days_passed
    
    projected_tax = summary['tax_liability']['total_tax'] * projection_factor
    quarterly_payment = projected_tax / 4
    
    print(f"\n💰 QUARTERLY TAX ESTIMATE")
    print(f"YTD Tax Liability: ${summary['tax_liability']['total_tax']:,.2f}")
    print(f"Projected Annual: ${projected_tax:,.2f}")
    print(f"Next Quarter Payment: ${quarterly_payment:,.2f}")
    
    # Find next due date
    for quarter, data in summary['quarterly_estimates'].items():
        if not data['paid']:
            print(f"Due Date: {data['due_date']}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tax reports for crypto trading")
    parser.add_argument('--year', type=int, help='Tax year (default: current year)')
    parser.add_argument('--output', type=str, default='reports/taxes', 
                       help='Output directory')
    parser.add_argument('--estimate', action='store_true', 
                       help='Estimate next quarterly payment')
    
    args = parser.parse_args()
    
    if args.estimate:
        estimate_next_quarter_payment()
    else:
        generate_tax_reports(args.year, args.output)