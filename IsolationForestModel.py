"""
Interactive Isolation Forest for NEO Anomaly Detection
Shows graph with matplotlib - hover to see NEO details
NOW SAVES MODEL TO JOBLIB FILE
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Database configuration
DB_PATH = "neo.db"

def load_neo_data():
    """Load NEO data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        name,
        neo_reference_id,
        absolute_magnitude_h,
        diameter_m,
        velocity_kms,
        miss_distance_km,
        hazardous,
        date
    FROM near_earth_objects
    WHERE diameter_m IS NOT NULL 
        AND velocity_kms IS NOT NULL 
        AND miss_distance_km IS NOT NULL
        AND absolute_magnitude_h IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def train_isolation_forest(df, contamination=0.1):
    """Train Isolation Forest model"""
    
    # Select features for anomaly detection
    feature_cols = ['absolute_magnitude_h', 'diameter_m', 'velocity_kms', 'miss_distance_km']
    X = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    # Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # Add predictions and scores to dataframe
    df['anomaly_prediction'] = predictions
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = (predictions == -1).astype(int)
    
    # ========== SAVE THE MODEL ==========
    print("\n" + "="*60)
    print("SAVING TRAINED MODELS...")
    print("="*60)
    
    # Save Isolation Forest model
    joblib.dump(iso_forest, 'neo_isolation_forest_model.joblib')
    print("✓ Isolation Forest model saved to: neo_isolation_forest_model.joblib")
    
    # Save the scaler
    joblib.dump(scaler, 'neo_isolation_forest_scaler.joblib')
    print("✓ Scaler saved to: neo_isolation_forest_scaler.joblib")
    
    print("="*60 + "\n")
    # ====================================
    
    return df, iso_forest, scaler, X_scaled

def create_interactive_plot(df, X_scaled):
    """Create matplotlib plot with hover annotations"""
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add PCA components to dataframe
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    # Separate normal and anomaly data
    normal_df = df[df['is_anomaly'] == 0]
    anomaly_df = df[df['is_anomaly'] == 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot normal objects (blue dots)
    scatter_normal = ax.scatter(normal_df['PC1'], normal_df['PC2'], 
               c='lightblue', s=30, alpha=0.5, 
               edgecolors='darkblue', linewidth=0.5,
               label=f'Normal ({len(normal_df):,})')
    
    # Plot anomalies (red X markers)
    scatter_anomaly = ax.scatter(anomaly_df['PC1'], anomaly_df['PC2'], 
                        c='red', s=120, alpha=0.8, 
                        marker='x', linewidths=2,
                        label=f'Anomaly ({len(anomaly_df):,})')
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Visualization of ALL 189,430 NEO Data Points', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Create annotation (hidden initially)
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="wheat", alpha=0.95),
                        arrowprops=dict(arrowstyle="->"),
                        fontsize=9, visible=False)
    
    # Hover event handler
    def hover(event):
        if event.inaxes == ax:
            # Check if mouse is over an anomaly point
            cont_anomaly, ind_anomaly = scatter_anomaly.contains(event)
            # Check if mouse is over a normal point
            cont_normal, ind_normal = scatter_normal.contains(event)
            
            if cont_anomaly:
                # Get the index of the hovered anomaly point
                idx = ind_anomaly["ind"][0]
                row = anomaly_df.iloc[idx]
                
                # Update annotation text with NEO characteristics
                text = (f"{row['name']}\n"
                       f"\n"
                       f"Characteristics:\n"
                       f"Diameter: {row['diameter_m']:.2f} m\n"
                       f"Velocity: {row['velocity_kms']:.2f} km/s\n"
                       f"Miss Distance: {row['miss_distance_km']:.2f} km\n"
                       f"Magnitude: {row['absolute_magnitude_h']:.2f}\n"
                       f"\n"
                       f"Hazardous: {'Yes' if row['hazardous'] == 1 else 'No'}\n"
                       f"Anomaly: Yes\n"
                       f"Anomaly Score: {row['anomaly_score']:.4f}\n"
                       f"Date: {row['date']}")
                
                annot.set_text(text)
                annot.xy = (row['PC1'], row['PC2'])
                annot.set_visible(True)
                fig.canvas.draw_idle()
                
            elif cont_normal:
                # Get the index of the hovered normal point
                idx = ind_normal["ind"][0]
                row = normal_df.iloc[idx]
                
                # Update annotation text with NEO characteristics
                text = (f"{row['name']}\n"
                       f"\n"
                       f"Characteristics:\n"
                       f"Diameter: {row['diameter_m']:.2f} m\n"
                       f"Velocity: {row['velocity_kms']:.2f} km/s\n"
                       f"Miss Distance: {row['miss_distance_km']:.2f} km\n"
                       f"Magnitude: {row['absolute_magnitude_h']:.2f}\n"
                       f"\n"
                       f"Hazardous: {'Yes' if row['hazardous'] == 1 else 'No'}\n"
                       f"Anomaly: No\n"
                       f"Anomaly Score: {row['anomaly_score']:.4f}\n"
                       f"Date: {row['date']}")
                
                annot.set_text(text)
                annot.xy = (row['PC1'], row['PC2'])
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    # Connect hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution - load data, train model, show graph"""
    
    print("="*60)
    print("NEO ISOLATION FOREST - Training and Saving Model")
    print("="*60 + "\n")
    
    # Load data
    print("Loading NEO data from database...")
    df = load_neo_data()
    print(f"✓ Loaded {len(df):,} NEO records\n")
    
    # Train model
    print("Training Isolation Forest model...")
    df, model, scaler, X_scaled = train_isolation_forest(df, contamination=0.02)
    print(f"✓ Model training complete")
    print(f"  - Found {(df['is_anomaly'] == 1).sum():,} anomalies ({(df['is_anomaly'] == 1).sum() / len(df) * 100:.1f}%)\n")
    
    # Create and show interactive plot
    print("Creating interactive visualization...")
    print("(Hover over points to see NEO details)\n")
    create_interactive_plot(df, X_scaled)

if __name__ == "__main__":
    main()