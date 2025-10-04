# Table Tennis RideShare Manager

An elegant PyQt6 desktop application that helps a table tennis team coordinate ride-sharing, calculate trip costs with real-world driving distances, and track outstanding balances between players. The interface embraces a modern palette of soft blues, crisp whites, and green accents, with rounded widgets, hover effects, and a responsive layout.

## ✨ Features

- **Google Maps integration** – Autocomplete start and destination addresses with the Places API and fetch accurate driving distances via the Distance Matrix API.
- **Ride cost calculator** – Combine flat driver fees with per-kilometre rates, split costs automatically across passengers, and handle validation edge cases gracefully.
- **Team management** – Add, rename, and remove players with persistent storage in SQLite, complete with validation, helpful feedback, and colour-coded rows that distinguish core members from reserves at a glance.
- **Ride history & ledger** – Review every saved trip, including passengers, costs, and the current balance of who owes whom.
- **Polished UI** – Styled with Qt Style Sheets for a clean, professional look featuring rounded corners, gradient buttons, and Segoe UI typography.
- **Smart defaults & history** – Store your preferred home address, fee defaults, and window size in `settings.json`, quickly reuse locations from past rides, and trim the history view to the three most recent trips while keeping a comprehensive net ledger.

## 📦 Requirements

- Python 3.10 or newer
- Google Maps API key with **Places API** and **Distance Matrix API** enabled
- Windows, macOS, or Linux desktop environment compatible with PyQt6

## 🚀 Quick Start

```powershell
git clone https://github.com/ChubbyChuckles/CarsharingApplication.git
cd CarsharingApplication
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```text
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

Launch the application:

```powershell
python -m src.main
```

On first launch the application seeds `%APPDATA%\TableTennisRideShare\rideshare.db` along with a matching `settings.json` file. When running from source those files are copied from `src/rideshare.db` and `src/config/settings.json` if present; otherwise they are initialised from the built-in defaults.
If the **Google Maps API key** is missing the application still opens, but you'll see an in-app warning and autocomplete/distance calculations remain disabled until the key is added.

## 🏗️ Build a Windows Installer (.msi)

The project ships with a `cx_Freeze` configuration that produces a native Windows installer.

```powershell
# from the project root
pip install -r requirements.txt
python scripts/build_msi.py bdist_msi
```

Or run the helper script (supports `-Clean` to wipe previous build output):

```powershell
.
\scripts\build_installer.ps1
```

Check the `dist/` directory for the generated `.msi` (for example, `TableTennisRideShareManager-0.1.0.msi`). Install it with a double-click or via `msiexec`. Place your `.env` file next to the installed `TableTennisRideShare.exe` (for example, `C:\Program Files\TableTennisRideShare\`) so the packaged app can find your Google Maps API key.

All user-modifiable data (the SQLite database and `settings.json`) now live in `%APPDATA%\TableTennisRideShare` to avoid permission issues under `Program Files`.

> The repository also includes an automated **Build and Publish MSI** GitHub Action (`.github/workflows/release-msi.yml`) that you can trigger manually or during a release to attach the installer to the GitHub release artifacts.

## 🧭 Using the App

- The interface now uses a frameless dark chrome with built-in minimize, maximize/restore, and close controls; you can drag the custom title bar or the window edges to reposition or resize it.

### Team Management Tab

- Add new teammates with the **Save Team Member** button and mark them as either **Core** or **Reserve** players.
- Select an existing row to rename, promote/demote between core and reserve, or delete a member. Members attached to past rides cannot be removed (it keeps the ledger consistent).

### Ride Setup Tab

- Start typing addresses to receive Google-powered autocomplete suggestions.
- Choose one or more drivers and any passengers taking part in the ride. Selected drivers are automatically excluded from the passenger list so they never contribute to the split.
- Enter each driver’s flat fee (in euros, applied per driver) alongside the per-kilometre rate. The app doubles the Google Maps driving distance to account for the return trip before calculating totals.
- Use the **Recent** drop-downs beside each address field to quickly reuse locations from previous trips, or rely on the default home address that pre-fills new rides.
- Click **Calculate Ride Cost** to see the round-trip distance, total cost, and the amount owed per core team member who actually rode. Core players who sit the ride out (or anyone marked reserve) simply stay unselected and are not charged.
- Click **Save Ride** to persist the trip; only core passengers are written to the ledger and debts are recorded in euros.

### Ride History & Ledger Tab

- Review a trimmed list of the three most recent rides with distances, fees, and participants, including every driver on multi-car trips. Older rides remain in the database and can still influence the ledger.
- Use the **Delete Selected Ride** button to remove an entry (and its associated ledger rows) if it was created in error.
- Inspect the live ledger, which now collapses all historical rides into net balances so you instantly see who ultimately owes whom (and by how much) across the entire season.
- Export the current ledger to a polished PDF with the **Export Ledger to PDF** button—perfect for sharing balances with the team.

## 🛠 Architecture Overview

| Component                                             | Responsibility                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `DatabaseManager`                                     | SQLite schema creation and CRUD for team members, rides, passengers, and ledger entries  |
| `GoogleMapsHandler`                                   | Thin wrapper around the `googlemaps` client for autocomplete and distance calculation    |
| `RideShareApp`                                        | Top-level `QMainWindow` hosting the tabbed interface                                     |
| `TeamManagementTab`, `RideSetupTab`, `RideHistoryTab` | Feature-specific widgets with high-level business logic                                  |
| `AddressLineEdit`                                     | Custom control that throttles autocomplete calls and feeds suggestions to a `QCompleter` |
| `resources/style.qss`                                 | Central stylesheet defining the modern visual design                                     |

All persistent data lives in `src/rideshare.db`. The schema uses foreign keys to maintain referential integrity and cascades deletions where appropriate.

## 🔑 Google Maps API Configuration

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Places API** and **Distance Matrix API**.
3. Generate an API key and restrict it to the enabled APIs for security.
4. Place the key in the `.env` file (or export `GOOGLE_MAPS_API_KEY` in your shell).

> ℹ️ The app exits gracefully with a helpful message if the key is missing or invalid.

## 🧪 Testing & Quality

- Run the automated test suite:

  ```powershell
  pytest
  ```

- Linting is configured via the tooling listed in `pyproject.toml`. Use your preferred formatter/linter before committing changes.

## 📁 Project Structure (excerpt)

```
src/
├── rideshare_app.py      # Main PyQt6 application with all GUI components
├── main.py               # Entry point delegating to rideshare_app.bootstrap_app
└── resources/
    └── style.qss         # Centralised QSS stylesheet for the UI
```

## 🤝 Contributing

1. Fork the repository and create a feature branch.
2. Ensure the application runs without errors and all tests pass.
3. Submit a pull request describing your changes and relevant screenshots where helpful.

## 📄 License

MIT License – see the [LICENSE](LICENSE) file for details.
