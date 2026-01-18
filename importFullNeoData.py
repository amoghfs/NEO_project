"""
Comprehensive NEO Data Fetcher
Fetches maximum historical data from NASA NeoWs API
"""

import requests
import sqlite3
import time
from datetime import datetime, timedelta

# NASA API configuration
API_KEY = "4znaUgLgFmi1vJanB3JG8h8I8zmL5mdQ2ZpIlQFO"
BASE_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# Database setup
DB_PATH = "neo.db"

def create_database():
    """Create database and table if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS near_earth_objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        name TEXT,
        neo_reference_id TEXT,
        absolute_magnitude_h REAL,
        diameter_min_m REAL,
        diameter_max_m REAL,
        diameter_m REAL,
        velocity_kmh REAL,
        velocity_kms REAL,
        miss_distance_km REAL,
        orbiting_body TEXT,
        hazardous INTEGER,
        timestamp TEXT,
        anomaly_score REAL,
        anomaly_label INTEGER,
        UNIQUE(neo_reference_id, date)
    )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized with duplicate prevention")

def fetch_neo_data(start_date, end_date):
    """Fetch NEO data for a date range (max 7 days per API call)"""
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check if any NEO data exists for this date range
        if data and "near_earth_objects" in data:
            total_neos = sum(len(neos) for neos in data["near_earth_objects"].values())
            if total_neos == 0:
                return {"empty": True, "message": "No NEO data for this date range"}
        
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return {"error": True, "message": f"Invalid date range"}
        return {"error": True, "message": str(e)}
    except requests.exceptions.RequestException as e:
        return {"error": True, "message": str(e)}

def parse_and_store_neos(data, fetch_timestamp):
    """Parse JSON and store in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stored_count = 0
    skipped_count = 0
    
    if not data or "near_earth_objects" not in data:
        conn.close()
        return stored_count, skipped_count
    
    for date, neos in data["near_earth_objects"].items():
        for neo in neos:
            neo_reference_id = neo["id"]
            name = neo["name"]
            abs_magnitude = neo.get("absolute_magnitude_h")
            
            # Get diameter
            diameter_data = neo.get("estimated_diameter", {}).get("meters", {})
            diameter_min = diameter_data.get("estimated_diameter_min")
            diameter_max = diameter_data.get("estimated_diameter_max")
            diameter_m = None
            if diameter_min and diameter_max:
                diameter_m = (float(diameter_min) + float(diameter_max)) / 2
            elif diameter_min:
                diameter_m = float(diameter_min)
            elif diameter_max:
                diameter_m = float(diameter_max)
            
            # Get close approach data
            close_approach = neo.get("close_approach_data", [])
            if not close_approach:
                continue
            
            approach = close_approach[0]
            velocity_kmh = approach.get("relative_velocity", {}).get("kilometers_per_hour")
            velocity_kms = approach.get("relative_velocity", {}).get("kilometers_per_second")
            miss_distance_km = approach.get("miss_distance", {}).get("kilometers")
            close_approach_date = approach.get("close_approach_date")
            orbiting_body = approach.get("orbiting_body", "Earth")
            
            # Convert to float
            if velocity_kmh:
                velocity_kmh = float(velocity_kmh)
            if velocity_kms:
                velocity_kms = float(velocity_kms)
            if miss_distance_km:
                miss_distance_km = float(miss_distance_km)
            
            hazardous = 1 if neo.get("is_potentially_hazardous_asteroid") else 0
            
            # Insert or skip if duplicate
            try:
                cursor.execute("""
                INSERT INTO near_earth_objects 
                (date, name, neo_reference_id, absolute_magnitude_h, 
                 diameter_min_m, diameter_max_m, diameter_m,
                 velocity_kmh, velocity_kms, miss_distance_km, 
                 orbiting_body, hazardous, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (close_approach_date, name, neo_reference_id, abs_magnitude,
                      diameter_data.get("estimated_diameter_min"), 
                      diameter_data.get("estimated_diameter_max"), 
                      diameter_m, velocity_kmh, velocity_kms, miss_distance_km,
                      orbiting_body, hazardous, fetch_timestamp))
                
                stored_count += 1
            except sqlite3.IntegrityError:
                skipped_count += 1
            except sqlite3.Error as e:
                print(f"Database error for NEO {neo_reference_id}: {e}")
    
    conn.commit()
    conn.close()
    
    return stored_count, skipped_count

def fetch_comprehensive_data(start_year=2015, end_year=2024):
    """Fetch comprehensive historical data from NASA NeoWs API"""
    
    # Validate date range
    if end_year < start_year:
        print("="*60)
        print("❌ ERROR: Invalid date range")
        print("="*60)
        print(f"Start year ({start_year}) must be BEFORE end year ({end_year})")
        print(f"Did you mean: start_year={end_year}, end_year={start_year}?")
        return
    
    if start_year < 1900:
        print("="*60)
        print("⚠️  WARNING: Date range very old")
        print("="*60)
        print(f"NASA's NEO data likely doesn't exist before 1900")
        print(f"Requested: {start_year}-{end_year}")
        print("Continuing anyway...\n")
    
    create_database()
    
    # Calculate date ranges
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current_date = start_date
    
    # Initialize counters
    total_stored = 0
    total_skipped = 0
    batch_count = 0
    no_data_count = 0
    error_count = 0
    
    print(f"Fetching NEO data from {start_date.date()} to {end_date.date()}")
    print("This will take a while due to API rate limits...")
    print("-" * 60)
    
    while current_date < end_date:
        # API allows max 7 days per request
        batch_end = min(current_date + timedelta(days=6), end_date)
        
        start_str = current_date.strftime("%Y-%m-%d")
        end_str = batch_end.strftime("%Y-%m-%d")
        
        print(f"Fetching: {start_str} to {end_str}...", end=" ")
        
        data = fetch_neo_data(start_str, end_str)
        
        if data and data.get("error"):
            print(f"✗ {data.get('message')}")
            error_count += 1
        elif data and data.get("empty"):
            print("⊘ No NEO data available")
            no_data_count += 1
        elif data:
            stored, skipped = parse_and_store_neos(data, datetime.now().strftime("%Y-%m-%d"))
            total_stored += stored
            total_skipped += skipped
            print(f"✓ Stored: {stored}, Skipped: {skipped}")
        else:
            print("✗ Failed")
            error_count += 1
        
        batch_count += 1
        current_date = batch_end + timedelta(days=1)
        
        # Respect API rate limits
        time.sleep(4)
        
        # Progress update every 50 batches
        if batch_count % 50 == 0:
            print(f"\n--- Progress: {batch_count} batches completed ---")
            print(f"Total NEOs stored: {total_stored}")
            print(f"Total duplicates skipped: {total_skipped}")
            print(f"Batches with no data: {no_data_count}")
            print(f"Batches with errors: {error_count}\n")
    
    print("\n" + "="*60)
    print("FETCH COMPLETE")
    print("="*60)
    print(f"Total batches processed: {batch_count}")
    print(f"NEOs stored: {total_stored}")
    print(f"Duplicates skipped: {total_skipped}")
    print(f"Batches with no data: {no_data_count}")
    print(f"Batches with errors: {error_count}")
    
    # Get statistics
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM near_earth_objects")
        total_records = cursor.fetchone()[0]
        
        if total_records == 0:
            print("\n" + "="*60)
            print("⚠️  NO DATA FOUND")
            print("="*60)
            print(f"No NEO data available for {start_year}-{end_year}.")
            print("\nPossible reasons:")
            print("  1. NASA's NeoWs API only has data from ~1900 onwards")
            print("  2. NEO tracking didn't exist before 1990s")
            print("  3. The date range is outside API coverage")
            print("\nRecommendation: Try years 2015-2024 for comprehensive data")
            conn.close()
            return
        
        cursor.execute("SELECT COUNT(*) FROM near_earth_objects WHERE hazardous = 1")
        hazardous_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM near_earth_objects WHERE hazardous = 0")
        non_hazardous_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM near_earth_objects")
        date_range = cursor.fetchone()
        
        conn.close()
        
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"Total unique NEOs: {total_records:,}")
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        
        hazard_pct = (hazardous_count / total_records * 100) if total_records > 0 else 0
        non_hazard_pct = (non_hazardous_count / total_records * 100) if total_records > 0 else 0
        
        print(f"Hazardous: {hazardous_count:,} ({hazard_pct:.2f}%)")
        print(f"Non-hazardous: {non_hazardous_count:,} ({non_hazard_pct:.2f}%)")
        
        if hazardous_count < 100:
            print("\n⚠️  WARNING: Less than 100 hazardous NEOs found")
            print("   Consider fetching more years for better model performance")
        elif hazardous_count < 500:
            print("\n⚠️  Consider fetching more data for better model performance")
        else:
            print("\n✓ Good amount of hazardous samples for training")
            
    except Exception as e:
        print(f"\n⚠️  Error calculating statistics: {e}")
        print("Data was stored successfully, but couldn't generate final report.")

if __name__ == "__main__":
    # Fetch data from desired year range
    # NASA NeoWs API has reliable data from ~2015 onwards
    fetch_comprehensive_data(start_year=2013, end_year=2014)