# TactiQ AI — Football Tactical Intelligence

A working React + TypeScript analytics prototype for AQX Sports Analytics Data Bowl 4.0.

## What works
- Responsive tactical dashboard with interactive match/team selectors
- Football pitch displaying completed forward passes
- Threat-zone overlay, passing success, progressive-pass counts
- Corridor-based passing comparisons and evidence-aware insights
- Player filter and CSV export
- Real StatsBomb event-data importer (separate script)

## Important methodological disclosure
- When first launched the app uses **synthetic demonstration passes** and displays a warning.
- The included spatial threat measure is **a handcrafted heuristic proxy, NOT a fitted or validated xT model**. It must not be called a calibrated probability of scoring.
- The network view aggregates completed player-to-recipient passes and uses average pass-origin positions (not tracking coordinates).
- Location grouping refers to the **pass start location**, from an attacking-left-to-right event coordinate frame (StatsBomb's event-location convention). This is not a verified player-position tracking visualization.
- Passing corridor comparisons are descriptive, not causal tactical prescriptions.
- Replace demo data using the importer before making competition claims.

## Run frontend (Windows PowerShell)
```powershell
cd frontend
npm install
npm run dev
```
Open the local URL Vite prints (normally http://localhost:5173).

## Load real matches
From the project root, with Python 3 installed and internet available:
```powershell
python analytics/import_statsbomb.py --limit 3
```
The script downloads a small number of freely available 2018 World Cup match event JSON files and writes `frontend/public/matches.json`. Refresh the frontend. Real matches are shown at the top of the selector.

## What's next for a competitive submission
1. Use a correctly documented, validated xT grid or train one with out-of-sample evaluation.
2. Validate passing-network connections and add minimum-sample filters.
3. Add match-held-out evaluation and confidence intervals, not just descriptive charts.
4. Verify calculations and publish a real-match example with source citations.
5. Add source attribution/logo and check StatsBomb conditions before deployment.

## Attribution
Football match event data is provided by **StatsBomb Open Data**: https://github.com/hudl/open-data. StatsBomb require attribution and their logo when publishing analyses using their data. Review their full terms and add their official logo to the published site/Devpost submission.

## License
Your original implementation may be licensed separately; StatsBomb source data remains governed by its own terms.
