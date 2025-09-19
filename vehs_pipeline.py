#!/usr/bin/env python3
"""
VEHS Data Cleaning and Enhancement Pipeline

This comprehensive pipeline processes EPCL VEHS Excel data to:
1. Clean and handle null values
2. Standardize date formats and field naming
3. Enrich data with additional context fields
4. Create relationships between records

Requirements:
- pandas
- openpyxl
- numpy
- datetime
- re
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class VEHSDataPipeline:
    """Comprehensive VEHS data cleaning and enhancement pipeline"""
    
    def __init__(self, excel_file_path: str):
        """Initialize pipeline with Excel file path"""
        self.excel_file_path = excel_file_path
        self.raw_data = {}
        self.cleaned_data = {}
        self.enriched_data = {}
        
        # Define standard field mappings and data types
        self.field_standardization = {
            'Date of Occurrence': 'occurrence_date',
            'Date Reported': 'reported_date',
            'Date Entered': 'entered_date',
            'Target Completion Date': 'target_completion_date',
            'Completion Date': 'completion_date',
            'Incident Number': 'incident_id',
            'Audit Number': 'audit_id',
            'Incident Type(s)': 'incident_type',
            'Group Company': 'company',
            'Location (EPCL)': 'location',
            'Department': 'department',
            'Sub-department': 'sub_department'
        }
        
        # Define expected date columns
        self.date_columns = [
            'occurrence_date', 'reported_date', 'entered_date',
            'target_completion_date', 'completion_date',
            'Entered Investigation', 'Entered Review', 'Entered Pending Closure',
            'Entered Closed', 'Start Date'
        ]
        
        # Define severity mappings
        self.severity_mapping = {
            'C0 - No Ill Effect': 0,
            'C1 - Minor': 1,
            'C2 - Serious': 2,
            'C3 - Severe': 3
        }
    
    def load_data(self):
        """Load all sheets from Excel file"""
        print("Loading Excel data...")
        try:
            excel_file = pd.ExcelFile(self.excel_file_path)
            for sheet_name in excel_file.sheet_names:
                print(f"  Loading sheet: {sheet_name}")
                self.raw_data[sheet_name] = pd.read_excel(
                    self.excel_file_path, 
                    sheet_name=sheet_name,
                    parse_dates=False  # We'll handle dates manually
                )
            print(f"Loaded {len(self.raw_data)} sheets successfully")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
    
    def clean_null_values(self):
        """Address high null values in key fields"""
        print("\nCleaning null values...")
        
        for sheet_name, df in self.raw_data.items():
            print(f"\nProcessing sheet: {sheet_name}")
            cleaned_df = df.copy()
            
            # Calculate null percentages for reporting
            null_percentages = (df.isnull().sum() / len(df)) * 100
            high_null_cols = null_percentages[null_percentages > 80].index.tolist()
            
            if high_null_cols:
                print(f"  High null columns (>80%): {len(high_null_cols)}")
            
            # Strategy 1: Drop columns with >95% nulls (likely unused fields)
            very_high_null_cols = null_percentages[null_percentages > 95].index.tolist()
            if very_high_null_cols:
                cleaned_df = cleaned_df.drop(columns=very_high_null_cols)
                print(f"  Dropped {len(very_high_null_cols)} columns with >95% nulls")
            
            # Strategy 2: Fill common categorical nulls
            categorical_fills = {
                'Status': 'Unknown',
                'Category': 'Not Specified',
                'Priority': 'Medium',
                'Department': 'Not Assigned',
                'Location': 'Not Specified',
                'Audit Status': 'Unknown'
            }
            
            for col, fill_value in categorical_fills.items():
                if col in cleaned_df.columns:
                    filled_count = cleaned_df[col].isnull().sum()
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    if filled_count > 0:
                        print(f"  Filled {filled_count} nulls in {col}")
            
            # Strategy 3: Forward fill for sequential data
            sequence_cols = ['Incident Number', 'Audit Number', 'Title', 'Description']
            for col in sequence_cols:
                if col in cleaned_df.columns:
                    before_nulls = cleaned_df[col].isnull().sum()
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                    after_nulls = cleaned_df[col].isnull().sum()
                    filled = before_nulls - after_nulls
                    if filled > 0:
                        print(f"  Forward filled {filled} nulls in {col}")
            
            # Strategy 4: Create null indicator columns for important fields
            important_cols = ['Root Cause', 'Corrective Actions', 'Investigation Team Leader']
            for col in important_cols:
                if col in cleaned_df.columns:
                    null_indicator = f"{col}_is_missing"
                    cleaned_df[null_indicator] = cleaned_df[col].isnull()
            
            self.cleaned_data[sheet_name] = cleaned_df
    
    def standardize_formats(self):
        """Standardize date formats and field naming"""
        print("\nStandardizing formats...")
        
        for sheet_name, df in self.cleaned_data.items():
            print(f"\nProcessing sheet: {sheet_name}")
            standardized_df = df.copy()
            
            # 1. Standardize column names
            rename_mapping = {}
            for old_name in standardized_df.columns:
                if old_name in self.field_standardization:
                    new_name = self.field_standardization[old_name]
                    rename_mapping[old_name] = new_name
                else:
                    # Clean column names: lowercase, replace spaces/special chars with underscores
                    clean_name = re.sub(r'[^\w\s]', '', old_name.lower())
                    clean_name = re.sub(r'\s+', '_', clean_name.strip())
                    clean_name = re.sub(r'_+', '_', clean_name)
                    if clean_name != old_name and clean_name not in rename_mapping.values():
                        rename_mapping[old_name] = clean_name
            
            if rename_mapping:
                standardized_df = standardized_df.rename(columns=rename_mapping)
                print(f"  Renamed {len(rename_mapping)} columns")
            
            # 2. Standardize date formats
            date_cols_found = []
            for col in standardized_df.columns:
                if any(date_keyword in col.lower() for date_keyword in ['date', 'entered', 'time']):
                    if standardized_df[col].dtype == 'object' or 'datetime' in str(standardized_df[col].dtype):
                        try:
                            standardized_df[col] = pd.to_datetime(standardized_df[col], errors='coerce')
                            date_cols_found.append(col)
                        except:
                            pass
            
            if date_cols_found:
                print(f"  Standardized {len(date_cols_found)} date columns")
            
            # 3. Standardize categorical values
            categorical_standardization = {
                'status': {'CLOSED': 'Closed', 'OPEN': 'Open', 'IN PROGRESS': 'In Progress'},
                'priority': {'HIGH': 'High', 'MEDIUM': 'Medium', 'LOW': 'Low'},
                'category': {'INCIDENT': 'Incident', 'HAZARD ID': 'Hazard ID', 'AUDIT': 'Audit'}
            }
            
            for col in standardized_df.columns:
                col_lower = col.lower()
                if col_lower in categorical_standardization:
                    mapping = categorical_standardization[col_lower]
                    standardized_df[col] = standardized_df[col].replace(mapping)
            
            # 4. Clean text fields
            text_columns = standardized_df.select_dtypes(include=['object']).columns
            for col in text_columns:
                try:
                    if col in standardized_df.columns and standardized_df[col].dtype == 'object':
                        # Remove extra whitespace and standardize
                        standardized_df[col] = standardized_df[col].astype(str).str.strip()
                        standardized_df[col] = standardized_df[col].replace('nan', np.nan)
                        standardized_df[col] = standardized_df[col].replace('None', np.nan)
                except (AttributeError, KeyError):
                    # Skip columns that cause issues
                    continue
            
            self.cleaned_data[sheet_name] = standardized_df
    
    def enrich_data(self):
        """Add context fields like cost impact, man-hours, risk scores"""
        print("\nEnriching data with additional context...")
        
        for sheet_name, df in self.cleaned_data.items():
            print(f"\nProcessing sheet: {sheet_name}")
            enriched_df = df.copy()
            
            # 1. Calculate severity scores
            if 'worst_case_consequence_incident' in enriched_df.columns:
                enriched_df['severity_score'] = enriched_df['worst_case_consequence_incident'].map(
                    self.severity_mapping
                ).fillna(1)  # Default to minor if unknown
            elif 'worst_case_consequence_potential_hazard_id' in enriched_df.columns:
                enriched_df['severity_score'] = enriched_df['worst_case_consequence_potential_hazard_id'].map(
                    self.severity_mapping
                ).fillna(1)
            
            # 2. Calculate time-based metrics
            if 'occurrence_date' in enriched_df.columns and 'reported_date' in enriched_df.columns:
                enriched_df['reporting_delay_days'] = (
                    enriched_df['reported_date'] - enriched_df['occurrence_date']
                ).dt.days.fillna(0)
            
            if 'reported_date' in enriched_df.columns and 'completion_date' in enriched_df.columns:
                enriched_df['resolution_time_days'] = (
                    enriched_df['completion_date'] - enriched_df['reported_date']
                ).dt.days
            
            # 3. Estimate cost impact based on severity and type
            cost_multipliers = {
                'Incident': {'base': 5000, 'multiplier': [1, 2, 5, 15]},  # C0-C3
                'Hazard ID': {'base': 1000, 'multiplier': [0.5, 1, 2, 5]},
                'Audit': {'base': 2000, 'multiplier': [1, 1.5, 3, 8]}
            }
            
            if 'category' in enriched_df.columns and 'severity_score' in enriched_df.columns:
                enriched_df['estimated_cost_impact'] = enriched_df.apply(
                    lambda row: self._estimate_cost(
                        row.get('category', 'Unknown'),
                        row.get('severity_score', 1),
                        cost_multipliers
                    ), axis=1
                )
            
            # 4. Estimate man-hours lost
            if 'severity_score' in enriched_df.columns:
                base_hours = {'Incident': 40, 'Hazard ID': 8, 'Audit': 16}
                enriched_df['estimated_manhours_impact'] = enriched_df.apply(
                    lambda row: base_hours.get(row.get('category', 'Incident'), 20) * 
                               (row.get('severity_score', 1) + 1), axis=1
                )
            
            # 5. Risk score calculation
            if 'severity_score' in enriched_df.columns:
                # Combine severity with other risk factors
                enriched_df['risk_score'] = enriched_df['severity_score'].fillna(1)
                
                # Increase risk for repeated incidents
                if 'repeated_incident' in enriched_df.columns:
                    mask = enriched_df['repeated_incident'].str.lower() == 'yes'
                    enriched_df.loc[mask, 'risk_score'] *= 1.5
                
                # Increase risk for overdue items
                if 'resolution_time_days' in enriched_df.columns:
                    overdue_mask = enriched_df['resolution_time_days'] > 30
                    enriched_df.loc[overdue_mask, 'risk_score'] *= 1.3
            
            # 6. Department risk profiling
            if 'department' in enriched_df.columns:
                dept_risk = enriched_df.groupby('department')['risk_score'].mean().to_dict()
                enriched_df['department_avg_risk'] = enriched_df['department'].map(dept_risk)
            
            # 7. Compliance indicators
            compliance_fields = ['management_system_non_compliance', 'psm', 'ems', 'ohih']
            compliance_counts = []
            for field in compliance_fields:
                if field in enriched_df.columns:
                    compliance_counts.append(enriched_df[field].notna().astype(int))
            
            if compliance_counts:
                # Sum across all compliance fields for each row
                enriched_df['compliance_systems_involved'] = pd.concat(compliance_counts, axis=1).sum(axis=1)
            else:
                enriched_df['compliance_systems_involved'] = 0
            
            self.enriched_data[sheet_name] = enriched_df
            print(f"  Added enrichment fields for {sheet_name}")
    
    def _estimate_cost(self, category: str, severity: float, cost_multipliers: dict) -> float:
        """Estimate cost impact based on category and severity"""
        if category not in cost_multipliers:
            category = 'Incident'  # Default
        
        base_cost = cost_multipliers[category]['base']
        severity_int = int(min(severity, 3))  # Cap at 3 (C3)
        multiplier = cost_multipliers[category]['multiplier'][severity_int]
        
        return base_cost * multiplier
    
    def create_relationships(self):
        """Link related records between incidents, audits, and corrective actions"""
        print("\nCreating relationships between records...")
        
        # Create master relationship table
        relationships = []
        
        # 1. Extract corrective action references from incidents
        if 'Incident' in self.enriched_data:
            incident_df = self.enriched_data['Incident']
            for idx, row in incident_df.iterrows():
                incident_id = row.get('incident_id')
                if pd.notna(incident_id):
                    # Look for corrective actions
                    corrective_actions = row.get('corrective_actions', '')
                    if pd.notna(corrective_actions):
                        # Extract action IDs (format: AC-XXX-YYYYMMDD-NNN)
                        action_ids = re.findall(r'AC-[A-Z]+-\d{8}-\d{3}', str(corrective_actions))
                        for action_id in action_ids:
                            relationships.append({
                                'source_type': 'Incident',
                                'source_id': incident_id,
                                'target_type': 'Corrective_Action',
                                'target_id': action_id,
                                'relationship_type': 'generates'
                            })
        
        # 2. Link audit findings to parent audits
        if 'Audit' in self.enriched_data and 'Audit Findings' in self.enriched_data:
            audit_df = self.enriched_data['Audit']
            findings_df = self.enriched_data['Audit Findings']
            
            for idx, finding in findings_df.iterrows():
                audit_id = finding.get('audit_id')
                if pd.notna(audit_id):
                    relationships.append({
                        'source_type': 'Audit',
                        'source_id': audit_id,
                        'target_type': 'Audit_Finding',
                        'target_id': f"{audit_id}_Finding_{idx}",
                        'relationship_type': 'contains'
                    })
        
        # 3. Link inspections to findings
        if 'Inspection' in self.enriched_data and 'Inspection Findings' in self.enriched_data:
            inspection_df = self.enriched_data['Inspection']
            findings_df = self.enriched_data['Inspection Findings']
            
            for idx, finding in findings_df.iterrows():
                audit_id = finding.get('audit_id')  # Inspection uses same ID field
                if pd.notna(audit_id):
                    relationships.append({
                        'source_type': 'Inspection',
                        'source_id': audit_id,
                        'target_type': 'Inspection_Finding',
                        'target_id': f"{audit_id}_Finding_{idx}",
                        'relationship_type': 'identifies'
                    })
        
        # 4. Create location-based relationships
        location_groups = {}
        for sheet_name, df in self.enriched_data.items():
            location_cols = [col for col in df.columns if 'location' in col.lower()]
            if location_cols:
                location_col = location_cols[0]  # Use first location column found
                try:
                    for location in df[location_col].dropna().unique():
                        if location not in location_groups:
                            location_groups[location] = []
                        location_groups[location].append(sheet_name)
                except (AttributeError, KeyError):
                    continue
        
        # 5. Create department-based relationships
        dept_groups = {}
        for sheet_name, df in self.enriched_data.items():
            dept_cols = [col for col in df.columns if 'department' in col.lower()]
            if dept_cols:
                dept_col = dept_cols[0]  # Use first department column found
                try:
                    for dept in df[dept_col].dropna().unique():
                        if dept not in dept_groups:
                            dept_groups[dept] = []
                        dept_groups[dept].append(sheet_name)
                except (AttributeError, KeyError):
                    continue
        
        # Store relationships
        if relationships:
            self.enriched_data['Relationships'] = pd.DataFrame(relationships)
            print(f"  Created {len(relationships)} relationship records")
        
        # Create summary tables
        self.enriched_data['Location_Summary'] = pd.DataFrame([
            {'location': loc, 'sheets_involved': len(sheets), 'sheet_list': ', '.join(sheets)}
            for loc, sheets in location_groups.items()
        ])
        
        self.enriched_data['Department_Summary'] = pd.DataFrame([
            {'department': dept, 'sheets_involved': len(sheets), 'sheet_list': ', '.join(sheets)}
            for dept, sheets in dept_groups.items()
        ])
        
        print(f"  Created location groups: {len(location_groups)}")
        print(f"  Created department groups: {len(dept_groups)}")
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\nGenerating data quality report...")
        
        report = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sheets_processed': len(self.enriched_data),
            'sheets': {}
        }
        
        for sheet_name, df in self.enriched_data.items():
            if sheet_name in ['Relationships', 'Location_Summary', 'Department_Summary']:
                continue
                
            sheet_report = {
                'rows_total': len(df),
                'columns_total': len(df.columns),
                'null_percentage': round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                'date_columns': len([col for col in df.columns if 'date' in col.lower()]),
                'enrichment_fields': len([col for col in df.columns if col in [
                    'severity_score', 'estimated_cost_impact', 'risk_score', 
                    'reporting_delay_days', 'resolution_time_days'
                ]]),
                'data_types': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
            }
            report['sheets'][sheet_name] = sheet_report
        
        return report
    
    def save_processed_data(self, output_path: str = None):
        """Save all processed data to Excel file"""
        if not output_path:
            base_name = os.path.splitext(self.excel_file_path)[0]
            output_path = f"{base_name}_processed.xlsx"
        
        print(f"\nSaving processed data to: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl', datetime_format='YYYY-MM-DD') as writer:
            for sheet_name, df in self.enriched_data.items():
                # Truncate sheet name if too long for Excel
                excel_sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                df.to_excel(writer, sheet_name=excel_sheet_name, index=False)
                print(f"  Saved sheet: {excel_sheet_name} ({len(df)} rows, {len(df.columns)} columns)")
        
        print(f"Successfully saved processed data with {len(self.enriched_data)} sheets")
    
    def run_pipeline(self, output_path: str = None):
        """Run the complete data pipeline"""
        print("=" * 60)
        print("VEHS DATA CLEANING AND ENHANCEMENT PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Clean null values
            self.clean_null_values()
            
            # Step 3: Standardize formats
            self.standardize_formats()
            
            # Step 4: Enrich data
            self.enrich_data()
            
            # Step 5: Create relationships
            self.create_relationships()
            
            # Step 6: Generate quality report
            quality_report = self.generate_data_quality_report()
            
            # Step 7: Save processed data
            self.save_processed_data(output_path)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Display summary
            print(f"Processed {quality_report['sheets_processed']} sheets")
            for sheet_name, stats in quality_report['sheets'].items():
                if sheet_name not in ['Relationships', 'Location_Summary', 'Department_Summary']:
                    print(f"  {sheet_name}: {stats['rows_total']} rows, {stats['columns_total']} columns, "
                          f"{stats['null_percentage']}% nulls, {stats['enrichment_fields']} enrichment fields")
            
            return quality_report
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function to run the pipeline"""
    # Configuration
    excel_file_path = "EPCL VEHS Data (Mar23 - Mar24).xlsx"
    output_path = "EPCL_VEHS_Data_Processed.xlsx"
    
    # Initialize and run pipeline
    pipeline = VEHSDataPipeline(excel_file_path)
    quality_report = pipeline.run_pipeline(output_path)
    
    # Save quality report
    report_path = "VEHS_Data_Quality_Report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print(f"\nData quality report saved to: {report_path}")
    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()