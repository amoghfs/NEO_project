"""
Setup and Validation Script for NEO Dashboard
Checks requirements, initializes database, and validates configuration
"""

import os
import sys
import sqlite3
from pathlib import Path

def check_python_version():
    """Ensure Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'requests',
        'sklearn',
        'xgboost',
        'joblib'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    models = {
        'xgboost_model.pkl': 'XGBoost classifier',
        'isolation_forest_model.pkl': 'Isolation Forest model',
        'scaler.pkl': 'Feature scaler (optional)'
    }
    
    all_found = True
    for filename, description in models.items():
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            if 'optional' in description:
                print(f"⚠️  {filename} - {description} - NOT FOUND (OK)")
            else:
                print(f"❌ {filename} - {description} - NOT FOUND")
                all_found = False
    
    if not all_found:
        print("\n⚠️  Missing required model files!")
        print("Train your models first using your training scripts")
        return False
    
    return True

def check_nasa_api_key():
    """Check if NASA API key is set"""
    print("\nChecking NASA API configuration...")
    api_key = os.getenv("NASA_API_KEY")
    
    if not api_key:
        print("⚠️  NASA_API_KEY not set - will use DEMO_KEY (30 requests/hour limit)")
        print("Get a free API key at: https://api.nasa.gov/")
        print("Set it with: export NASA_API_KEY='your_key_here'")
        return True  # Not critical, can use DEMO_KEY
    
    if api_key == "DEMO_KEY":
        print("⚠️  Using DEMO_KEY (30 requests/hour limit)")
        print("Get a free API key at: https://api.nasa.gov/")
    else:
        print(f"✅ NASA_API_KEY is set (starts with: {api_key[:10]}...)")
    
    return True

def init_database():
    """Initialize the SQLite database"""
    print("\nInitializing database...")
    db_path = "neo.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Create predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS neo_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    name TEXT,
                    neo_reference_id TEXT UNIQUE,
                    orbiting_body TEXT,
                    diameter_m REAL,
                    miss_distance_km REAL,
                    velocity_kmh REAL,
                    hazardous INTEGER,
                    absolute_magnitude REAL,
                    xgb_risk_prob REAL,
                    isolation_anomaly_score REAL,
                    is_anomaly INTEGER,
                    risk_score REAL,
                    risk_label TEXT,
                    prediction_time_utc TEXT
                )
            """)
            
            # Check if data exists
            cursor = conn.execute("SELECT COUNT(*) FROM neo_predictions")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"✅ Database initialized with {count:,} existing records")
            else:
                print("✅ Database initialized (empty)")
                print("   Run realtime_neo_updater.py to populate with data")
            
            conn.commit()
        
        return True
    
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free // (2**30)
        print(f"✅ Free disk space: {free_gb} GB")
        
        if free_gb < 1:
            print("⚠️  Warning: Less than 1 GB free space")
            print("   Recommended: At least 5 GB for historical data")
            return False
        
        return True
    
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Not critical

def test_database_write():
    """Test database write permissions"""
    print("\nTesting database write access...")
    db_path = "neo.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Try to write a test record
            test_id = "test_neo_001"
            conn.execute(
                "INSERT OR REPLACE INTO neo_predictions (neo_reference_id, name, risk_score) VALUES (?, ?, ?)",
                (test_id, "Test NEO", 0.5)
            )
            
            # Try to read it back
            cursor = conn.execute(
                "SELECT name, risk_score FROM neo_predictions WHERE neo_reference_id = ?",
                (test_id,)
            )
            result = cursor.fetchone()
            
            # Clean up
            conn.execute("DELETE FROM neo_predictions WHERE neo_reference_id = ?", (test_id,))
            conn.commit()
            
            if result and result[0] == "Test NEO":
                print("✅ Database write test successful")
                return True
            else:
                print("❌ Database write test failed")
                return False
    
    except Exception as e:
        print(f"❌ Database write test failed: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    
    directories = [
        '.streamlit'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ {dir_name}/")
    
    return True

def print_next_steps(all_checks_passed):
    """Print instructions for next steps"""
    print("\n" + "="*60)
    
    if all_checks_passed:
        print("🎉 Setup complete! You're ready to go!")
        print("\nNext steps:")
        print("\n1. Start the data updater (in one terminal):")
        print("   python realtime_neo_updater.py")
        print("\n2. Start the dashboard (in another terminal):")
        print("   streamlit run enhanced_dashboard.py")
        print("\n3. Open your browser to: http://localhost:8501")
        print("\n4. For deployment, see DEPLOYMENT_README.md")
    else:
        print("⚠️  Setup incomplete - please fix the issues above")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Train and save your ML models")
        print("- Get NASA API key: https://api.nasa.gov/")
    
    print("="*60 + "\n")

def main():
    """Run all setup checks"""
    print("="*60)
    print("NEO Dashboard Setup & Validation")
    print("="*60)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model files", check_model_files),
        ("NASA API key", check_nasa_api_key),
        ("Disk space", check_disk_space),
        ("Directory structure", create_directory_structure),
        ("Database initialization", init_database),
        ("Database write access", test_database_write),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append(False)
    
    all_passed = all(results)
    print_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
