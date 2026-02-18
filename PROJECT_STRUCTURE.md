# 🗂️ AIS Collision Risk MVP - Project Structure

## File Organization

```
Marine Life for Upload/
│
├── Whale_tracking.py                    # UPDATED - Main Streamlit app
├── image_ai_page.py                     # EXISTING - Don't touch
├── iot_buoy_page.py                     # EXISTING - Don't touch
├── train_whale_model.py                 # EXISTING - Don't touch
├── model.ipynb                          # EXISTING - Don't touch
│
├── ais_collision_page.py                # NEW - AIS collision risk page
├── ais_utils.py                         # NEW - AIS data fetching utilities
├── risk_calculator.py                   # NEW - Risk calculation logic
│
├── config.json                          # NEW - Configuration file
│
├── Blue whales Eastern North Pacific... # EXISTING - Whale data
├── whales_only.csv                      # EXISTING - Whale data
├── SetupGuide.txt                       # EXISTING
│
└── __pycache__/                         # Auto-generated
```

## New Files to Create

### 1. `config.json` (Configuration)
- AIS API credentials
- Geographic bounds
- Risk thresholds

### 2. `ais_utils.py` (AIS Data Fetching)
- Fetch vessel data from AISHub API
- Parse and clean AIS data
- Filter high-risk vessels

### 3. `risk_calculator.py` (Risk Calculation)
- Calculate distance between vessels and whales
- Compute risk scores
- Categorize risk levels

### 4. `ais_collision_page.py` (Streamlit Page)
- Main AIS dashboard
- Interactive map
- Risk alerts table

### 5. `Whale_tracking.py` (UPDATED)
- Add new page option to sidebar
- Import and call AIS page

### 6. `SETUP_GUIDE.md` (Setup Instructions)
- Installation steps
- API configuration
- Running the app

## Files NOT to Modify

- ❌ `image_ai_page.py`
- ❌ `iot_buoy_page.py`
- ❌ `train_whale_model.py`
- ❌ `model.ipynb`
- ❌ CSV data files

## Installation Order

1. Create `config.json`
2. Create `ais_utils.py`
3. Create `risk_calculator.py`
4. Create `ais_collision_page.py`
5. Update `Whale_tracking.py`
6. Install dependencies
7. Configure API credentials
8. Run application

---

**Total New Files:** 5 (4 new + 1 updated)
**Lines of Code:** ~800 lines total
**Time to Setup:** 30 minutes
