"""
PatrolIQ - Step 1: Data Acquisition and Preprocessing
This script downloads, samples, and preprocesses the Chicago crime dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CrimeDataPreprocessor:
    def __init__(self, sample_size=500000):
        self.sample_size = sample_size
        self.df = None
        
    def load_data(self, filepath=None):
        """
        Load Chicago crime dataset
        If filepath is None, it will download from the Chicago Data Portal
        """
        print("Loading Chicago Crime Dataset...")
        
        if filepath:
            # Load from local file
            self.df = pd.read_csv(filepath)
        else:
            # Download from Chicago Data Portal - Fixed URL
            # Using proper URL encoding and simpler parameter format
            url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit=500000"
            print("Downloading data from Chicago Data Portal...")
            try:
                self.df = pd.read_csv(url)
            except Exception as e:
                print(f"Error downloading data: {e}")
                print("Trying alternative method...")
                # Alternative: Use smaller limit if needed
                url_alt = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit=100000"
                self.df = pd.read_csv(url_alt)
        
        print(f"Dataset loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        return self.df
    
    def sample_data(self):
        """Sample recent crime records"""
        if len(self.df) > self.sample_size:
            print(f"Sampling {self.sample_size} recent records...")
            # Sort by date first if date column exists
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                self.df = self.df.sort_values('date', ascending=False)
            self.df = self.df.head(self.sample_size)
        print(f"Working with {len(self.df)} records")
        return self.df
    
    def clean_data(self):
        """Comprehensive data cleaning"""
        print("Starting data cleaning...")
        
        initial_rows = len(self.df)
        
        # Standardize column names (Chicago portal uses lowercase)
        self.df.columns = self.df.columns.str.title().str.replace(' ', '_')
        
        # Handle both possible column name formats
        id_col = 'Id' if 'Id' in self.df.columns else 'ID'
        lat_col = 'Latitude' if 'Latitude' in self.df.columns else 'latitude'
        lon_col = 'Longitude' if 'Longitude' in self.df.columns else 'longitude'
        type_col = 'Primary_Type' if 'Primary_Type' in self.df.columns else 'primary_type'
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        
        # Remove duplicates
        if id_col in self.df.columns:
            self.df.drop_duplicates(subset=[id_col], keep='first', inplace=True)
            print(f"Removed {initial_rows - len(self.df)} duplicate records")
        
        # Handle missing values in critical columns
        critical_cols = []
        if lat_col in self.df.columns:
            critical_cols.append(lat_col)
        if lon_col in self.df.columns:
            critical_cols.append(lon_col)
        if type_col in self.df.columns:
            critical_cols.append(type_col)
        if date_col in self.df.columns:
            critical_cols.append(date_col)
            
        if critical_cols:
            self.df.dropna(subset=critical_cols, inplace=True)
            print(f"Removed records with missing critical data. Remaining: {len(self.df)}")
        
        # Fill missing values in other columns
        if 'Arrest' in self.df.columns:
            self.df['Arrest'].fillna(False, inplace=True)
            self.df['Arrest'] = self.df['Arrest'].astype(bool)
            
        if 'Domestic' in self.df.columns:
            self.df['Domestic'].fillna(False, inplace=True)
            self.df['Domestic'] = self.df['Domestic'].astype(bool)
            
        if 'District' in self.df.columns:
            self.df['District'].fillna(0, inplace=True)
            
        if 'Ward' in self.df.columns:
            self.df['Ward'].fillna(0, inplace=True)
            
        if 'Community_Area' in self.df.columns:
            self.df['Community_Area'].fillna(0, inplace=True)
        
        print("Data cleaning completed!")
        return self.df
    
    def extract_temporal_features(self):
        """Extract temporal features from datetime"""
        print("Extracting temporal features...")
        
        # Find the date column
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        
        # Convert Date to datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
        
        # Extract temporal components
        self.df['Hour'] = self.df[date_col].dt.hour
        self.df['Day_Of_Week'] = self.df[date_col].dt.day_name()
        self.df['Month'] = self.df[date_col].dt.month
        self.df['Year'] = self.df[date_col].dt.year
        self.df['Day'] = self.df[date_col].dt.day
        
        # Create weekend flag
        self.df['Is_Weekend'] = self.df['Day_Of_Week'].isin(['Saturday', 'Sunday']).astype(int)
        
        # Create season
        def get_season(month):
            if pd.isna(month):
                return 'Unknown'
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        self.df['Season'] = self.df['Month'].apply(get_season)
        
        # Rename Date column for consistency
        if date_col != 'Date':
            self.df.rename(columns={date_col: 'Date'}, inplace=True)
        
        print("Temporal features extracted successfully!")
        return self.df
    
    def create_crime_severity_score(self):
        """Create crime severity scores based on crime type"""
        print("Creating crime severity scores...")
        
        # Find the primary type column
        type_col = 'Primary_Type' if 'Primary_Type' in self.df.columns else 'primary_type'
        if type_col not in self.df.columns:
            print("Warning: Primary_Type column not found, skipping severity scores")
            return self.df
        
        # Rename for consistency
        if type_col != 'Primary_Type':
            self.df.rename(columns={type_col: 'Primary_Type'}, inplace=True)
        
        # Define severity levels (higher = more severe)
        severity_mapping = {
            'HOMICIDE': 10,
            'CRIM SEXUAL ASSAULT': 9,
            'KIDNAPPING': 9,
            'ROBBERY': 8,
            'ASSAULT': 7,
            'BATTERY': 6,
            'BURGLARY': 6,
            'ARSON': 7,
            'WEAPONS VIOLATION': 7,
            'MOTOR VEHICLE THEFT': 5,
            'THEFT': 4,
            'CRIMINAL DAMAGE': 4,
            'NARCOTICS': 5,
            'CRIMINAL TRESPASS': 3,
            'DECEPTIVE PRACTICE': 3,
            'INTERFERENCE WITH PUBLIC OFFICER': 5,
            'INTIMIDATION': 5,
            'STALKING': 6,
            'SEX OFFENSE': 8,
            'OFFENSE INVOLVING CHILDREN': 8,
            'PUBLIC PEACE VIOLATION': 2,
            'GAMBLING': 2,
            'LIQUOR LAW VIOLATION': 2,
            'OBSCENITY': 3
        }
        
        self.df['Crime_Severity_Score'] = self.df['Primary_Type'].map(severity_mapping)
        self.df['Crime_Severity_Score'].fillna(3, inplace=True)  # Default medium severity
        
        print("Crime severity scores created!")
        return self.df
    
    def data_quality_report(self):
        """Generate data quality report"""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Records: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns)}")
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print(missing.head(10))
        else:
            print("No missing values!")
        
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        
        print("\nNumeric Columns Summary:")
        print(self.df.describe())
        
        print("\n" + "="*60)
        
    def save_processed_data(self, filename='processed_crime_data.csv'):
        """Save processed data"""
        self.df.to_csv(filename, index=False)
        print(f"\nProcessed data saved to {filename}")
        
    def run_preprocessing_pipeline(self, filepath=None):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        try:
            # Step 1: Load data
            self.load_data(filepath)
            
            # Step 2: Sample data
            self.sample_data()
            
            # Step 3: Clean data
            self.clean_data()
            
            # Step 4: Extract temporal features
            self.extract_temporal_features()
            
            # Step 5: Create severity scores
            self.create_crime_severity_score()
            
            # Step 6: Quality report
            self.data_quality_report()
            
            # Step 7: Save processed data
            self.save_processed_data()
            
            print("\n" + "="*60)
            print("PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\nError in preprocessing pipeline: {e}")
            print("Please check your internet connection or try loading from a local file.")
            raise
        
        return self.df


# Example Usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CrimeDataPreprocessor(sample_size=500000)
    
    # Load from your local file
    filepath = r'C:\project\PatrolIQ  project\data\PatrolIQ_Chicago_Crime_500K.csv'
    df = preprocessor.run_preprocessing_pipeline(filepath)
    
    # Display sample data
    print("\nSample Processed Data:")
    print(df.head())
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    print("\nCrime Types Distribution:")
    if 'Primary_Type' in df.columns:
        print(df['Primary_Type'].value_counts().head(10))