#!/usr/bin/env python3
"""
Excel Sheet, Column, and Data Extractor

This script extracts all sheet names, their corresponding column names,
and the first 5 rows of data from an Excel file and displays them in a structured format.

Requirements:
- pandas
- openpyxl

Usage:
    python extract_excel_info.py
"""

import pandas as pd
import os
from typing import Dict, List, Tuple


def extract_excel_info(file_path: str) -> Dict[str, Tuple[List[str], pd.DataFrame]]:
    """
    Extract sheet names, column names, and first 5 rows from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        Dict[str, Tuple[List[str], pd.DataFrame]]: Dictionary with sheet names as keys 
        and tuples of (column_names, first_5_rows_dataframe) as values
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Dictionary to store sheet names and their data
    excel_info = {}
    
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in the Excel file:")
        print("=" * 80)
        
        # Extract column names and first 5 rows for each sheet
        for sheet_name in sheet_names:
            try:
                # Read the first 5 rows to get both column names and data
                df_sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                column_names = df_sample.columns.tolist()
                excel_info[sheet_name] = (column_names, df_sample)
                
                print(f"\nSheet: '{sheet_name}'")
                print(f"Number of columns: {len(column_names)}")
                print("Columns:")
                for i, col in enumerate(column_names, 1):
                    print(f"  {i:2d}. {col}")
                
                print(f"\nFirst 5 rows of data:")
                if not df_sample.empty:
                    # Display the dataframe with better formatting
                    print(df_sample.to_string(index=True, max_cols=None, max_colwidth=20))
                else:
                    print("  No data found in this sheet")
                
                print("-" * 80)
                    
            except Exception as e:
                print(f"\nError reading sheet '{sheet_name}': {str(e)}")
                excel_info[sheet_name] = ([], pd.DataFrame())
                
    except Exception as e:
        raise Exception(f"Error processing Excel file: {str(e)}")
    
    return excel_info


def save_to_text_file(excel_info: Dict[str, Tuple[List[str], pd.DataFrame]], output_file: str):
    """
    Save the extracted information to a text file.
    
    Args:
        excel_info (Dict[str, Tuple[List[str], pd.DataFrame]]): Dictionary with sheet and data information
        output_file (str): Path to the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Excel File Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        for sheet_name, (columns, df_sample) in excel_info.items():
            f.write(f"Sheet: {sheet_name}\n")
            f.write(f"Number of columns: {len(columns)}\n")
            f.write("Columns:\n")
            
            if columns:
                for i, col in enumerate(columns, 1):
                    f.write(f"  {i:2d}. {col}\n")
            else:
                f.write("  No columns found or error reading sheet\n")
            
            f.write(f"\nFirst 5 rows of data:\n")
            if not df_sample.empty:
                # Convert dataframe to string with better formatting for text file
                df_string = df_sample.to_string(index=True, max_cols=None, max_colwidth=30)
                f.write(df_string)
                f.write("\n")
            else:
                f.write("No data found in this sheet\n")
            
            f.write("\n" + "=" * 80 + "\n\n")


def save_to_csv_files(excel_info: Dict[str, Tuple[List[str], pd.DataFrame]], output_dir: str = "extracted_data"):
    """
    Save the first 5 rows of each sheet to separate CSV files.
    
    Args:
        excel_info (Dict[str, Tuple[List[str], pd.DataFrame]]): Dictionary with sheet and data information
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nSaving sample data to CSV files in '{output_dir}' directory...")
    
    for sheet_name, (columns, df_sample) in excel_info.items():
        if not df_sample.empty:
            # Clean sheet name for filename (remove invalid characters)
            clean_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            csv_filename = f"{clean_name}_first_5_rows.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            try:
                df_sample.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"  Saved: {csv_filename}")
            except Exception as e:
                print(f"  Error saving {csv_filename}: {str(e)}")


def save_to_markdown(excel_info: Dict[str, Tuple[List[str], pd.DataFrame]], output_file: str = "excel_data_analysis.md"):
    """
    Save the extracted information to a Markdown file optimized for AI analysis.
    
    Args:
        excel_info (Dict[str, Tuple[List[str], pd.DataFrame]]): Dictionary with sheet and data information
        output_file (str): Path to the output Markdown file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Excel Data Analysis Report\n\n")
        f.write("This document contains the structure and sample data from all sheets in the Excel file.\n\n")
        
        # Table of contents
        f.write("## Table of Contents\n\n")
        for i, sheet_name in enumerate(excel_info.keys(), 1):
            f.write(f"{i}. [{sheet_name}](#{sheet_name.lower().replace(' ', '-').replace('(', '').replace(')', '')})\n")
        f.write("\n---\n\n")
        
        for sheet_name, (columns, df_sample) in excel_info.items():
            # Sheet header
            f.write(f"## {sheet_name}\n\n")
            
            if columns:
                f.write(f"**Total Columns:** {len(columns)}\n")
                f.write(f"**Sample Rows:** {len(df_sample)}\n\n")
                
                # Data types analysis
                if not df_sample.empty:
                    f.write("### Data Types\n\n")
                    f.write("| Column | Data Type | Non-Null Count | Sample Values |\n")
                    f.write("|--------|-----------|----------------|---------------|\n")
                    
                    for col in columns:
                        if col in df_sample.columns:
                            dtype = str(df_sample[col].dtype)
                            non_null = df_sample[col].count()
                            # Get first few non-null values as sample
                            sample_vals = df_sample[col].dropna().head(3).tolist()
                            sample_str = ", ".join([str(v)[:20] + "..." if len(str(v)) > 20 else str(v) for v in sample_vals])
                            f.write(f"| {col} | {dtype} | {non_null}/{len(df_sample)} | {sample_str} |\n")
                        else:
                            f.write(f"| {col} | Unknown | 0 | N/A |\n")
                    f.write("\n")
                
                # Sample data table
                f.write("### Sample Data (First 5 Rows)\n\n")
                if not df_sample.empty:
                    # Convert DataFrame to Markdown table
                    md_table = df_sample.to_markdown(index=False, tablefmt='github')
                    f.write(md_table)
                    f.write("\n\n")
                else:
                    f.write("*No data available in this sheet*\n\n")
                    
            else:
                f.write("*Error reading this sheet or no columns found*\n\n")
            
            f.write("---\n\n")
        
        # Summary section
        f.write("## Summary\n\n")
        total_sheets = len(excel_info)
        sheets_with_data = sum(1 for _, df in excel_info.values() if not df[1].empty)
        total_columns = sum(len(columns) for columns, _ in excel_info.values())
        
        f.write(f"- **Total Sheets:** {total_sheets}\n")
        f.write(f"- **Sheets with Data:** {sheets_with_data}\n")
        f.write(f"- **Total Columns:** {total_columns}\n")
        f.write(f"- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### Sheet Overview\n\n")
        f.write("| Sheet Name | Columns | Sample Rows | Has Data |\n")
        f.write("|------------|---------|-------------|----------|\n")
        for sheet_name, (columns, df_sample) in excel_info.items():
            has_data = "✅" if not df_sample.empty else "❌"
            f.write(f"| {sheet_name} | {len(columns)} | {len(df_sample)} | {has_data} |\n")


def main():
    """Main function to execute the script."""
    # Excel file path
    excel_file_path = "EPCL_VEHS_Data_Processed.xlsx"
    
    # Output file for saving results
    output_file_path = "excel_analysis_report.txt"
    
    try:
        print("Extracting Excel file information and sample data...")
        print(f"File: {excel_file_path}")
        print("=" * 80)
        
        # Extract information
        excel_info = extract_excel_info(excel_file_path)
        
        # Save to text file
        save_to_text_file(excel_info, output_file_path)
        
        # Save sample data to CSV files
        save_to_csv_files(excel_info)
        
        # Save to Markdown format (AI-friendly)
        markdown_file = "excel_data_analysis.md"
        save_to_markdown(excel_info, markdown_file)
        
        print("\n" + "=" * 80)
        print(f"Analysis complete!")
        print(f"Text report saved to: {output_file_path}")
        print(f"AI-friendly Markdown report saved to: {markdown_file}")
        print(f"Sample data saved to individual CSV files in 'extracted_data' directory")
        print(f"Total sheets processed: {len(excel_info)}")
        
        # Summary
        total_columns = sum(len(columns) for columns, _ in excel_info.values())
        sheets_with_data = sum(1 for _, df in excel_info.values() if not df[1].empty)
        print(f"Total columns across all sheets: {total_columns}")
        print(f"Sheets with data: {sheets_with_data}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()