"""
Analyze annual results from EnergyPlus simulation.
Extracts key metrics from the output files.
"""

import os
import sys
import pandas as pd


def analyze_csv_results(output_dir):
    """Analyze the CSV output file for annual totals."""
    csv_path = os.path.join(output_dir, 'eplusout.csv')
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    print("="*70)
    print("ANNUAL ENERGY ANALYSIS FROM CSV")
    print("="*70 + "\n")
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"Simulation period: {len(df)} timesteps")
        print(f"Columns available: {len(df.columns)}")
        
        # Show available columns
        print("\nAvailable data columns:")
        for i, col in enumerate(df.columns[:20], 1):  # Show first 20
            print(f"  {i}. {col}")
        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more columns")
        
        # Calculate annual totals for energy columns
        print("\n" + "="*70)
        print("ANNUAL TOTALS (sum of all timesteps)")
        print("="*70)
        
        energy_keywords = ['Energy', 'Electricity', 'Gas', 'Heating', 'Cooling']
        
        for col in df.columns:
            if any(keyword in col for keyword in energy_keywords):
                total = df[col].sum()
                if total != 0:  # Only show non-zero values
                    print(f"\n{col}:")
                    print(f"  Annual Total: {total:,.2f}")
                    print(f"  Average: {df[col].mean():,.2f}")
                    print(f"  Peak: {df[col].max():,.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def analyze_meter_results(output_dir):
    """Analyze the meter output file."""
    mtr_path = os.path.join(output_dir, 'eplusout.mtr')
    
    if not os.path.exists(mtr_path):
        print(f"\nMeter file not found: {mtr_path}")
        return
    
    print("\n" + "="*70)
    print("METER DATA ANALYSIS")
    print("="*70 + "\n")
    
    try:
        # Read meter file (similar format to CSV)
        df = pd.read_csv(mtr_path)
        
        print(f"Meter data timesteps: {len(df)}")
        print(f"Meters available: {len(df.columns)}")
        
        # Calculate totals
        print("\nAnnual Meter Totals:")
        for col in df.columns:
            if col != 'Date/Time':
                total = df[col].sum()
                if total != 0:
                    print(f"  {col}: {total:,.2f}")
        
    except Exception as e:
        print(f"Error reading meter file: {e}")


def show_html_report_location(output_dir):
    """Show where to find the HTML report."""
    html_path = os.path.join(output_dir, 'eplustbl.htm')
    
    print("\n" + "="*70)
    print("DETAILED ANNUAL REPORT")
    print("="*70)
    
    if os.path.exists(html_path):
        print(f"\n✓ HTML Report: {os.path.abspath(html_path)}")
        print("\nThis report contains:")
        print("  • Annual Building Utility Performance")
        print("  • End Uses by Category")
        print("  • Monthly Energy Consumption")
        print("  • Peak Demand Analysis")
        print("  • Comfort Summary")
        print("  • HVAC Sizing Summary")
        print("  • And much more...")
        
        # Try to open it
        try:
            import webbrowser
            open_now = input("\nOpen HTML report in browser? (y/n): ").strip().lower()
            if open_now == 'y':
                webbrowser.open(f"file:///{os.path.abspath(html_path)}")
                print("✓ Opened in browser")
        except:
            pass
    else:
        print(f"\n✗ HTML report not found at: {html_path}")


def main():
    """Main analysis function."""
    # Default to small office output
    output_dir = "outputs/small_office"
    
    # Allow custom output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("\nUsage:")
        print("  python analyze_annual_results.py [output_directory]")
        print("\nExample:")
        print("  python analyze_annual_results.py outputs/small_office")
        return
    
    print("="*70)
    print("ENERGYPLUS ANNUAL RESULTS ANALYZER")
    print("="*70)
    print(f"\nAnalyzing: {os.path.abspath(output_dir)}\n")
    
    # Analyze CSV results
    df = analyze_csv_results(output_dir)
    
    # Analyze meter results
    analyze_meter_results(output_dir)
    
    # Show HTML report location
    show_html_report_location(output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
