import requests
import sqlite3
from datetime import datetime

# === NASA API CONFIG ===
API_KEY = "4znaUgLgFmi1vJanB3JG8h8I8zmL5mdQ2ZpIlQFO"
START_DATE = "2023-07-13"
END_DATE = "2023-07-18" 
URL = (
    f"https://api.nasa.gov/neo/rest/v1/feed"
    f"?start_date={START_DATE}&end_date={END_DATE}&api_key={API_KEY}"
)

# === DATABASE CONFIG ===
DB_FILE = "neo.db"               
TABLE_NAME = "near_earth_objects"

try:
    # === Call the API ===
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()

    all_neos = []
    hazardous_count = 0

    # === Iterate over NEOs by date ===
    for date, neos in data["near_earth_objects"].items():
        for neo in neos:
            # Skip if there is no close_approach_data (just to be safe)
            if not neo.get("close_approach_data"):
                continue

            cad = neo["close_approach_data"][0]

            name = neo["name"]
            neo_reference_id = neo["neo_reference_id"]
            hazardous = bool(neo["is_potentially_hazardous_asteroid"])

            # Absolute magnitude H (brightness proxy)
            absolute_magnitude_h = float(neo["absolute_magnitude_h"])

            # Diameters in meters
            diameter_min_m = float(
                neo["estimated_diameter"]["meters"]["estimated_diameter_min"]
            )
            diameter_max_m = float(
                neo["estimated_diameter"]["meters"]["estimated_diameter_max"]
            )

            # Use max as a simple single diameter feature
            diameter_m = diameter_max_m

            # Velocity: API gives string, kilometers per hour
            velocity_kmh = float(
                cad["relative_velocity"]["kilometers_per_hour"]
            )
            velocity_kms = velocity_kmh / 3600.0  # convert to km/s

            # Miss distance in kilometers
            miss_distance_km = float(cad["miss_distance"]["kilometers"])

            orbiting_body = cad.get("orbiting_body", None)

            record = {
                "date": date,  # close-approach date from outer loop
                "name": name,
                "neo_reference_id": neo_reference_id,
                "absolute_magnitude_h": absolute_magnitude_h,
                "diameter_min_m": diameter_min_m,
                "diameter_max_m": diameter_max_m,
                "diameter_m": diameter_m,
                "velocity_kmh": velocity_kmh,
                "velocity_kms": velocity_kms,
                "miss_distance_km": miss_distance_km,
                "orbiting_body": orbiting_body,
                "hazardous": 1 if hazardous else 0,  # store as integer
                "timestamp": datetime.utcnow().isoformat(),
            }
            all_neos.append(record)

            if hazardous:
                hazardous_count += 1

    # === Print all NEOs at the end (for debugging) ===
    print("Near-Earth Objects details:\n")
    for neo in all_neos[:10]:  # only first 10 to avoid huge output
        print(f"🌑 Name: {neo['name']}")
        print(f"   ▪ Date: {neo['date']}")
        print(f"   ▪ H (abs mag): {neo['absolute_magnitude_h']:.2f}")
        print(f"   ▪ Diameter max: {neo['diameter_max_m']:.2f} m")
        print(f"   ▪ Velocity: {neo['velocity_kmh']:.2f} km/h ({neo['velocity_kms']:.3f} km/s)")
        print(f"   ▪ Miss Distance: {neo['miss_distance_km']:.2f} km")
        print(f"   ▪ Orbiting body: {neo['orbiting_body']}")
        print(f"   ▪ Hazardous: {'Yes' if neo['hazardous'] else 'No'}\n")

    total_neos = len(all_neos)
    print(f"Total Near-Earth Objects detected: {total_neos}")
    print(f"Total Hazardous NEOs detected: {hazardous_count}\n")

    # === SAVE TO DATABASE ===
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
            timestamp TEXT
        )
    """)

    # Optional: unique index to avoid exact duplicates for same NEO & date
    cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date_neo
        ON {TABLE_NAME} (date, neo_reference_id);
    """)

    # Insert all records (ignore duplicates if re-running for same dates)
    for neo in all_neos:
        cursor.execute(f"""
            INSERT OR IGNORE INTO {TABLE_NAME}
            (date, name, neo_reference_id,
             absolute_magnitude_h,
             diameter_min_m, diameter_max_m, diameter_m,
             velocity_kmh, velocity_kms,
             miss_distance_km, orbiting_body,
             hazardous, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            neo["date"],
            neo["name"],
            neo["neo_reference_id"],
            neo["absolute_magnitude_h"],
            neo["diameter_min_m"],
            neo["diameter_max_m"],
            neo["diameter_m"],
            neo["velocity_kmh"],
            neo["velocity_kms"],
            neo["miss_distance_km"],
            neo["orbiting_body"],
            neo["hazardous"],
            neo["timestamp"],
        ))

    conn.commit()
    conn.close()
    print(f"💾 All data saved to '{DB_FILE}' in table '{TABLE_NAME}'")

except requests.exceptions.RequestException as e:
    print("❌ Error fetching data:", e)
except Exception as e:
    print("❌ Error saving data:", e)
