# 🚀 AIS Collision Risk MVP - Quick Reference Card

## Instant Setup (30 Minutes)

### 1. Install Packages (2 minutes)
```bash
pip install requests geopy
```

### 2. Get AISHub API (5 minutes)
- Visit: http://www.aishub.net/
- Register (free)
- Note your username

### 3. Configure (2 minutes)
Edit `config.json` line 4:
```json
"username": "your_aishub_username"
```

### 4. Run Application (1 minute)
```bash
streamlit run Whale_tracking.py
```

### 5. Use the Feature
- Select "AIS Collision Risk" from sidebar
- Click "🔄 Fetch Live Data"
- View results!

---

## File Locations

All 4 new files go in the same folder as `Whale_tracking.py`:

```
✅ config.json
✅ ais_utils.py
✅ risk_calculator.py
✅ ais_collision_page.py
✅ Whale_tracking.py (REPLACE with new version)
```

---

## Risk Level Colors

| Color | Risk Level | Distance | Action |
|-------|-----------|----------|--------|
| 🔴 Red | CRITICAL | < 2 km | Immediate alert |
| 🟠 Orange | HIGH | 2-5 km | Contact vessel |
| 🟡 Yellow | MEDIUM | 5-15 km | Monitor closely |
| 🟢 Green | LOW | 15-30 km | Continue monitoring |

---

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| "No vessels found" | Try different area with ship traffic |
| "Configure username" | Edit config.json with AISHub username |
| Module not found | `pip install requests geopy` |
| API limit exceeded | Wait 1 hour (60 requests/hour limit) |

---

## Geographic Bounds Quick Reference

**California Coast:**
```
Lat: 32 to 42
Lon: -125 to -117
```

**East Coast USA:**
```
Lat: 35 to 45
Lon: -78 to -65
```

**English Channel:**
```
Lat: 49 to 52
Lon: -5 to 2
```

---

## Important Limits

- **API Calls**: 60 per hour (free tier)
- **Recommended**: 1 call every 5 minutes
- **Data Latency**: 1-5 minutes
- **Coverage**: Global (where AIS available)

---

## Test Your Setup

```bash
# Check packages
python -c "import requests, geopy; print('✅ Packages OK')"

# Check files
ls config.json ais_*.py risk_*.py

# Run app
streamlit run Whale_tracking.py
```

---

## Quick Help

**Can't find vessels?**
→ Use California coast bounds (32-42, -125 to -117)

**API error?**
→ Check username in config.json

**No whale zones?**
→ Verify CSV file is in same folder

---

## Success Checklist

- [ ] Packages installed
- [ ] AISHub account created  
- [ ] config.json configured
- [ ] All 5 files in place
- [ ] App runs without errors
- [ ] Can fetch vessel data
- [ ] Map displays correctly
- [ ] Risk alerts show up

---

**Need detailed help?** See SETUP_GUIDE.md

**Ready to start?** Run: `streamlit run Whale_tracking.py`
