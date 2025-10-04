# Table Tennis RideShare Manager

An elegant PyQt6 desktop application that helps a table tennis team coordinate ride-sharing, calculate trip costs with real-world driving distances, and track outstanding balances between players. The interface embraces a modern palette of soft blues, crisp whites, and green accents, with rounded widgets, hover effects, and a responsive layout.

## ✨ Features

- **Google Maps integration** – Autocomplete start and destination addresses with the Places API and fetch accurate driving distances via the Distance Matrix API.
- **Ride cost calculator** – Combine flat driver fees with per-kilometre rates, split costs automatically across passengers, and handle validation edge cases gracefully.
- **Team management** – Add, rename, and remove players with persistent storage in SQLite, complete with validation and helpful feedback.
- **Ride history & ledger** – Review every saved trip, including passengers, costs, and the current balance of who owes whom.
- **Polished UI** – Styled with Qt Style Sheets for a clean, professional look featuring rounded corners, gradient buttons, and Segoe UI typography.

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

Create a `.env` file in the project root (same folder as `pyproject.toml`) and add your Google key:

```text
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

Launch the application:

```powershell
python -m src.main
```

On first launch the database file `src/rideshare.db` is created automatically with the required tables.

## 🧭 Using the App

### Team Management Tab

- Add new teammates with the **Save Team Member** button and mark them as either **Core** or **Reserve** players.
- Select an existing row to rename, promote/demote between core and reserve, or delete a member. Members attached to past rides cannot be removed (it keeps the ledger consistent).

### Ride Setup Tab

- Start typing addresses to receive Google-powered autocomplete suggestions.
- Choose a driver and one or more passengers (the driver is automatically excluded from the passenger list).
- Enter the driver’s flat fee (in euros) and per-kilometre rate. The app doubles the Google Maps driving distance to account for the return trip before calculating totals.
- Click **Calculate Ride Cost** to see the round-trip distance, total cost, and the amount owed per core team member. Reserve players are listed but never charged.
- Click **Save Ride** to persist the trip; only core passengers are written to the ledger and debts are recorded in euros.

### Ride History & Ledger Tab

- Review a chronological list of rides with distances, fees, and participants.
- See a live ledger of outstanding amounts per passenger/driver pair.

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
