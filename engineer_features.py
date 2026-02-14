import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CrimeFeatureEngineer:
    def __init__(self, processed_data_path=None):
        """
        Initialize Feature Engineer
        
        Args:
            processed_data_path: Path to the processed crime data CSV file
        """
        self.df = None
        self.processed_data_path = processed_data_path or r'C:\project\PatrolIQ  project\data_preprocessing\processed_crime_data.csv'
        
    def load_processed_data(self):
        """Load preprocessed crime data"""
        print("="*70)
        print("SCRIPT 03: FEATURE ENGINEERING")
        print("="*70)
        print(f"\n[1/8] Loading processed data from: {self.processed_data_path}")
        
        try:
            self.df = pd.read_csv(self.processed_data_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} records, {self.df.shape[1]} features")
            return self.df
        except FileNotFoundError:
            print(f"✗ Processed data not found at: {self.processed_data_path}")
            print("Please run: python scripts/02_preprocess_data.py")
            raise
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def create_spatial_features(self):
        """Create spatial density and clustering features"""
        print("\n[2/8] Creating spatial features...")
        
        # Check if required columns exist
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            print("⚠ Warning: Latitude/Longitude columns not found, skipping spatial features")
            return self.df
        
        # Grid-based spatial features (divide city into grid cells)
        lat_bins = 50
        lon_bins = 50
        
        self.df['Lat_Grid'] = pd.cut(self.df['Latitude'], bins=lat_bins, labels=False)
        self.df['Lon_Grid'] = pd.cut(self.df['Longitude'], bins=lon_bins, labels=False)
        
        # Create grid cell identifier
        self.df['Grid_Cell'] = (self.df['Lat_Grid'].astype(str) + '_' + 
                                self.df['Lon_Grid'].astype(str))
        
        # Calculate crime density per grid cell
        grid_crime_count = self.df.groupby('Grid_Cell').size().reset_index(name='Grid_Crime_Count')
        self.df = self.df.merge(grid_crime_count, on='Grid_Cell', how='left')
        
        # Distance from city center (approximate Chicago center)
        chicago_center_lat = 41.8781
        chicago_center_lon = -87.6298
        
        self.df['Distance_From_Center'] = np.sqrt(
            (self.df['Latitude'] - chicago_center_lat)**2 + 
            (self.df['Longitude'] - chicago_center_lon)**2
        )
        
        print(f"✓ Spatial features created: Grid_Cell, Grid_Crime_Count, Distance_From_Center")
        return self.df
    
    def create_temporal_aggregations(self):
        """Create temporal aggregation features"""
        print("\n[3/8] Creating temporal aggregation features...")
        
        if 'Date' not in self.df.columns:
            print("⚠ Warning: Date column not found, skipping temporal aggregations")
            return self.df
        
        # Ensure Date is datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Sort by date for rolling calculations
        self.df.sort_values('Date', inplace=True)
        
        # Hour-based aggregations
        if 'Hour' in self.df.columns:
            hour_crime_count = self.df.groupby('Hour').size().reset_index(name='Hourly_Crime_Count')
            self.df = self.df.merge(hour_crime_count, on='Hour', how='left')
        
        # Day of week aggregations
        if 'Day_Of_Week' in self.df.columns:
            dow_crime_count = self.df.groupby('Day_Of_Week').size().reset_index(name='DOW_Crime_Count')
            self.df = self.df.merge(dow_crime_count, on='Day_Of_Week', how='left')
        
        # Month aggregations
        if 'Month' in self.df.columns:
            month_crime_count = self.df.groupby('Month').size().reset_index(name='Monthly_Crime_Count')
            self.df = self.df.merge(month_crime_count, on='Month', how='left')
        
        print(f"✓ Temporal aggregations created")
        return self.df
    
    def create_crime_type_features(self):
        """Create features based on crime types"""
        print("\n[4/8] Creating crime type features...")
        
        if 'Primary_Type' not in self.df.columns:
            print("⚠ Warning: Primary_Type column not found, skipping crime type features")
            return self.df
        
        # Crime type frequency
        crime_type_count = self.df.groupby('Primary_Type').size().reset_index(name='Crime_Type_Frequency')
        self.df = self.df.merge(crime_type_count, on='Primary_Type', how='left')
        
        # Create binary flags for major crime categories
        violent_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 
                         'CRIM SEXUAL ASSAULT', 'KIDNAPPING']
        property_crimes = ['THEFT', 'BURGLARY', 'MOTOR VEHICLE THEFT', 
                          'CRIMINAL DAMAGE', 'ARSON']
        
        self.df['Is_Violent_Crime'] = self.df['Primary_Type'].isin(violent_crimes).astype(int)
        self.df['Is_Property_Crime'] = self.df['Primary_Type'].isin(property_crimes).astype(int)
        
        print(f"✓ Crime type features created: Crime_Type_Frequency, Is_Violent_Crime, Is_Property_Crime")
        return self.df
    
    def create_location_features(self):
        """Create features based on location"""
        print("\n[5/8] Creating location-based features...")
        
        # District-based features
        if 'District' in self.df.columns:
            district_crime_count = self.df.groupby('District').size().reset_index(name='District_Crime_Count')
            self.df = self.df.merge(district_crime_count, on='District', how='left')
        
        # Ward-based features
        if 'Ward' in self.df.columns:
            ward_crime_count = self.df.groupby('Ward').size().reset_index(name='Ward_Crime_Count')
            self.df = self.df.merge(ward_crime_count, on='Ward', how='left')
        
        # Community Area features
        if 'Community_Area' in self.df.columns:
            community_crime_count = self.df.groupby('Community_Area').size().reset_index(name='Community_Crime_Count')
            self.df = self.df.merge(community_crime_count, on='Community_Area', how='left')
        
        # Location Description features (if available)
        if 'Location_Description' in self.df.columns:
            loc_desc_count = self.df.groupby('Location_Description').size().reset_index(name='Location_Desc_Crime_Count')
            self.df = self.df.merge(loc_desc_count, on='Location_Description', how='left')
        
        print(f"✓ Location features created")
        return self.df
    
    def create_arrest_features(self):
        """Create arrest-related features"""
        print("\n[6/8] Creating arrest features...")
        
        if 'Arrest' not in self.df.columns:
            print("⚠ Warning: Arrest column not found, skipping arrest features")
            return self.df
        
        # Arrest rate by crime type
        if 'Primary_Type' in self.df.columns:
            arrest_rate = self.df.groupby('Primary_Type')['Arrest'].mean().reset_index(name='Crime_Type_Arrest_Rate')
            self.df = self.df.merge(arrest_rate, on='Primary_Type', how='left')
        
        # Arrest rate by district
        if 'District' in self.df.columns:
            district_arrest_rate = self.df.groupby('District')['Arrest'].mean().reset_index(name='District_Arrest_Rate')
            self.df = self.df.merge(district_arrest_rate, on='District', how='left')
        
        print(f"✓ Arrest features created")
        return self.df
    
    def create_time_based_features(self):
        """Create advanced time-based features"""
        print("\n[7/8] Creating advanced time-based features...")
        
        if 'Hour' not in self.df.columns:
            print("⚠ Warning: Hour column not found, skipping time-based features")
            return self.df
        
        # Time of day categories
        def categorize_time(hour):
            if pd.isna(hour):
                return 'Unknown'
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        
        self.df['Time_Of_Day'] = self.df['Hour'].apply(categorize_time)
        
        # Peak hours flag (6-9 AM and 4-7 PM)
        self.df['Is_Peak_Hour'] = self.df['Hour'].apply(
            lambda x: 1 if (6 <= x <= 9 or 16 <= x <= 19) else 0
        )
        
        # Late night flag (10 PM - 4 AM)
        self.df['Is_Late_Night'] = self.df['Hour'].apply(
            lambda x: 1 if (x >= 22 or x <= 4) else 0
        )
        
        print(f"✓ Time-based features created: Time_Of_Day, Is_Peak_Hour, Is_Late_Night")
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\n[8/8] Encoding categorical features...")
        
        # List of categorical columns to encode
        categorical_cols = []
        
        for col in ['Primary_Type', 'Day_Of_Week', 'Season', 'Time_Of_Day', 
                   'Location_Description', 'Description']:
            if col in self.df.columns:
                categorical_cols.append(col)
        
        if not categorical_cols:
            print("⚠ No categorical columns found to encode")
            return self.df
        
        # Label encoding for categorical variables
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle missing values
            self.df[col] = self.df[col].fillna('Unknown')
            self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col].astype(str))
            label_encoders[col] = le
        
        print(f"✓ Encoded {len(categorical_cols)} categorical features")
        return self.df
    
    def feature_summary(self):
        """Generate feature engineering summary"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*70)
        
        print(f"\nFinal Dataset Shape: {self.df.shape}")
        print(f"Total Features: {len(self.df.columns)}")
        
        # List all new features
        print("\nFeatures Created:")
        feature_categories = {
            'Spatial': [col for col in self.df.columns if any(x in col for x in ['Grid', 'Distance', 'Lat_', 'Lon_'])],
            'Temporal': [col for col in self.df.columns if any(x in col for x in ['Hourly', 'DOW_', 'Monthly', 'Time_Of', 'Peak', 'Late_Night'])],
            'Crime Type': [col for col in self.df.columns if any(x in col for x in ['Crime_Type', 'Violent', 'Property', 'Severity'])],
            'Location': [col for col in self.df.columns if any(x in col for x in ['District_', 'Ward_', 'Community_', 'Location_Desc'])],
            'Arrest': [col for col in self.df.columns if 'Arrest_Rate' in col],
            'Encoded': [col for col in self.df.columns if col.endswith('_Encoded')]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"\n{category} Features ({len(features)}):")
                for feat in features[:5]:  # Show first 5
                    print(f"  • {feat}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
        
        print("\n" + "="*70)
    
    def save_engineered_data(self, filename='engineered_crime_data.csv'):
        """Save engineered dataset"""
        output_path = filename
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Engineered data saved to: {output_path}")
        print(f"  Records: {len(self.df):,}")
        print(f"  Features: {len(self.df.columns)}")
    
    def run_feature_engineering_pipeline(self):
        """Run complete feature engineering pipeline"""
        try:
            # Step 1: Load processed data
            self.load_processed_data()
            
            # Step 2: Create spatial features
            self.create_spatial_features()
            
            # Step 3: Create temporal aggregations
            self.create_temporal_aggregations()
            
            # Step 4: Create crime type features
            self.create_crime_type_features()
            
            # Step 5: Create location features
            self.create_location_features()
            
            # Step 6: Create arrest features
            self.create_arrest_features()
            
            # Step 7: Create time-based features
            self.create_time_based_features()
            
            # Step 8: Encode categorical features
            self.encode_categorical_features()
            
            # Summary
            self.feature_summary()
            
            # Save engineered data
            self.save_engineered_data()
            
            print("\n" + "="*70)
            print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
            print("="*70 + "\n")
            
            return self.df
            
        except Exception as e:
            print(f"\n✗ Error in feature engineering pipeline: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    # Initialize feature engineer with your processed data path
    processed_data_path = r'C:\project\PatrolIQ  project\data_preprocessing\processed_crime_data.csv'
    
    engineer = CrimeFeatureEngineer(processed_data_path=processed_data_path)
    
    # Run feature engineering pipeline
    df_engineered = engineer.run_feature_engineering_pipeline()
    
    # Display sample of engineered data
    print("\nSample Engineered Data:")
    print(df_engineered.head())
    
    print("\nAll Features:")

    print(df_engineered.columns.tolist())
