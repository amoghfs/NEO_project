# NEO Risk Dashboard - Deployment Guide

## Overview
Real-time Near-Earth Object risk assessment dashboard using XGBoost and Isolation Forest models.

## Project Structure
```
.
├── enhanced_dashboard.py          # Streamlit dashboard
├── realtime_neo_updater.py       # Background data fetcher & predictor
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment process definitions
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── neo.db                        # SQLite database (auto-created)
├── xgboost_model.pkl            # Your trained XGBoost model
├── isolation_forest_model.pkl   # Your trained Isolation Forest model
└── scaler.pkl                   # Feature scaler (if using)
```

## Setup Instructions

### 1. Local Development

**Prerequisites:**
- Python 3.8+
- Your trained ML models (XGBoost, Isolation Forest)
- NASA API key (get from https://api.nasa.gov/)

**Steps:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set NASA API key as environment variable
export NASA_API_KEY="your_api_key_here"

# Run the data updater (in one terminal)
python realtime_neo_updater.py

# Run the dashboard (in another terminal)
streamlit run enhanced_dashboard.py
```

### 2. Deployment Options

#### Option A: Streamlit Cloud (Recommended for Dashboard)
**Best for:** Free hosting, easy deployment, public access

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Set environment variable: `NASA_API_KEY`
5. Deploy!

**Note:** Streamlit Cloud doesn't support background workers. For real-time updates, use Option B or C.

#### Option B: Railway.app (Best for Full System)
**Best for:** Both dashboard + background updater, $5/month budget

1. Create account at https://railway.app/
2. Create new project from GitHub repo
3. Add environment variables:
   ```
   NASA_API_KEY=your_key_here
   ```
4. Railway will automatically:
   - Detect Procfile
   - Run both web (dashboard) and worker (updater) processes
   - Provide public URL

**Database:** Railway provides persistent disk storage for SQLite.

#### Option C: Heroku
**Best for:** Reliable hosting, free tier available

1. Install Heroku CLI
2. Create Heroku app:
   ```bash
   heroku create your-neo-dashboard
   ```
3. Set environment variables:
   ```bash
   heroku config:set NASA_API_KEY=your_key_here
   ```
4. Deploy:
   ```bash
   git push heroku main
   ```
5. Scale dynos:
   ```bash
   heroku ps:scale web=1 worker=1
   ```

**Database:** Use Heroku's persistent filesystem or upgrade to Postgres.

#### Option D: PythonAnywhere
**Best for:** Simple deployment, free tier

1. Upload files to PythonAnywhere
2. Set up virtual environment
3. Configure web app to run `enhanced_dashboard.py`
4. Set up scheduled task for `realtime_neo_updater.py`

### 3. Environment Variables

Required:
- `NASA_API_KEY`: Your NASA API key (default: "DEMO_KEY" with 30 requests/hour limit)

Optional:
- `PORT`: Server port (auto-set by most hosting platforms)

### 4. Database Setup

The SQLite database (`neo.db`) is automatically created on first run.

**Initial Population:**
- If you have historical data (1975-2025), import it first
- The updater will start fetching from the most recent date in the database
- If starting fresh, it begins from today

**To import historical data:**
```python
import sqlite3
import pandas as pd

# Load your historical data
df_historical = pd.read_csv("your_historical_data.csv")

# Save to database
with sqlite3.connect("neo.db") as conn:
    df_historical.to_sql("neo_predictions", conn, if_exists="append", index=False)
```

### 5. Model Files

Ensure these files are present:
- `xgboost_model.pkl` - Your trained XGBoost classifier
- `isolation_forest_model.pkl` - Your trained Isolation Forest model
- `scaler.pkl` - (Optional) Feature scaler

**Important:** Make sure model features match the features defined in `realtime_neo_updater.py`

### 6. Cost Breakdown (for $10/month budget)

**Railway.app:**
- Starter plan: $5/month
- 512MB RAM, shared CPU
- 1GB storage (sufficient for ~2-5M NEO records)
- Unlimited bandwidth

**Hosting alternatives:**
- Streamlit Cloud: Free (dashboard only, no background worker)
- Heroku: Free tier (550 hours/month for hobby apps)
- PythonAnywhere: $5/month
- DigitalOcean: $6/month (requires more setup)

### 7. API Rate Limits

NASA NeoWs API limits:
- **With API key**: 1,000 requests/hour
- **DEMO_KEY**: 30 requests/hour, 50 requests/day

The updater is configured to stay well within these limits:
- Max 1,000 requests/day
- Max 10 requests/minute
- Automatic rate limiting and retry logic

### 8. Testing Locally

```bash
# Terminal 1: Start updater
python realtime_neo_updater.py

# Terminal 2: Start dashboard
streamlit run enhanced_dashboard.py

# Open browser to: http://localhost:8501
```

### 9. Monitoring

**Check if updater is running:**
- View logs in hosting platform dashboard
- Check database update timestamps
- Monitor `last_update` field in dashboard

**Performance metrics:**
- Dashboard should load in < 2 seconds
- Predictions should update within 5 seconds of new data
- Should handle 50+ concurrent users

### 10. Troubleshooting

**Dashboard shows no data:**
- Run `realtime_neo_updater.py` to populate database
- Check if `neo.db` exists and contains data
- Verify model files are present

**API rate limit errors:**
- Get a proper API key from https://api.nasa.gov/
- Don't use DEMO_KEY in production
- Check logs for rate limit messages

**Models not loading:**
- Ensure model files are in the same directory
- Check file paths in `realtime_neo_updater.py`
- Verify model files are compatible with current scikit-learn version

**Database errors:**
- Check file permissions for `neo.db`
- Ensure sufficient disk space
- Use absolute paths if relative paths fail

### 11. Customization

**Update interval:**
Edit `UPDATE_INTERVAL` in `realtime_neo_updater.py` (default: 300 seconds)

**Auto-refresh rate:**
Edit `AUTO_REFRESH_SECONDS` in `enhanced_dashboard.py` (default: 30 seconds)

**Risk score weights:**
Adjust in `NEOPredictor.predict()`:
```python
df["risk_score"] = (
    0.6 * df["xgb_risk_prob"] +      # XGBoost weight
    0.4 * df["isolation_anomaly_score"]  # Isolation Forest weight
)
```

**Risk thresholds:**
Adjust in `enhanced_dashboard.py`:
```python
df["risk_label"] = pd.cut(
    df["risk_score"],
    bins=[0, 0.3, 0.7, 1.0],  # Modify these thresholds
    labels=["LOW", "MEDIUM", "HIGH"]
)
```

### 12. Domain Setup (Optional)

To use a custom domain:

**Railway:**
1. Go to project settings
2. Add custom domain
3. Update DNS records

**Heroku:**
```bash
heroku domains:add www.your-neo-dashboard.com
```

Popular domain registrars:
- Namecheap: ~$10/year for .com
- Google Domains: ~$12/year
- Cloudflare: At-cost pricing (~$9/year)

### 13. Security Considerations

- Never commit `NASA_API_KEY` to Git
- Use environment variables for sensitive data
- Enable HTTPS (automatic on most platforms)
- Keep dependencies updated: `pip install -U -r requirements.txt`

### 14. Maintenance

**Weekly:**
- Check dashboard uptime
- Review error logs
- Verify data is updating

**Monthly:**
- Update Python packages
- Review API usage
- Check disk space usage

**As needed:**
- Retrain models with new data
- Adjust risk thresholds based on testing
- Optimize database queries if slow

## Support

For project-specific questions, refer to your research plan documentation.

For technical issues:
- Streamlit: https://docs.streamlit.io/
- Railway: https://docs.railway.app/
- NASA API: https://api.nasa.gov/

---

**Good luck with your engineering project! 🚀**
