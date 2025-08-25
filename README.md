# Matrix Cargo

Intelligent tracking and prediction system for cargo logistics.

## Overview

Matrix Cargo is a for route optimization, real-time vehicle tracking, and delivery time prediction for logistics operations. The system uses machine learning and geospatial data analysis to provide accurate travel time estimates and detect anomalies in cargo transportation.

> **Note:** The demo script was generated using AI, and some Matrix Tracking functionalities are currently simulations only.

## Key Features

- **Route Planning**: Optimized route calculation based on geographic data.
- **Real-Time Tracking**: Continuous monitoring of vehicle positions.
- **Duration Prediction**: Accurate travel time estimation using machine learning models.
- **Anomaly Detection**: Automatic identification of route deviations and unexpected delays.
- **Cluster Analysis**: Intelligent grouping of origin and destination locations to improve predictions.

## Project Structure

```
matrixcargo/
├── demo.py                      # System demonstration script
├── models/                      # Machine learning models
│   ├── experiments/             # Experimental models
│   └── production/              # Production models
├── notebooks/                   # Jupyter notebooks for analysis and exploration
└── src/                         # Project source code
    ├── features/                # Feature engineering
    ├── matrix_tracking/         # Tracking system
    ├── pipeline/                # Data processing pipeline
    └── predict/                 # Prediction modules
```

## Technologies Used

- **Python**: Main development language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Haversine**: Geographic distance calculations
- **Jupyter**: Exploratory analysis and visualization

## Installation and Setup

To set up and run Matrix Cargo, follow these instructions:

1. Clone the repository:
   ```bash
   git clone https://github.com/theohillmann/matrix-cargo.git
   cd matrix-cargo
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify the models in the production folder:
   Make sure the following files are present in the `models/production/` folder:
   - `end_cluster_model.pkl`
   - `start_cluster_model.pkl`
   - `ronsomForestRefressor.pkl`
   
   If they are not present, copy them to the mentioned folder.

5. Run the demo:
   ```bash
   python demo.py
   ```

## How to Use

### Demonstration

Run the demonstration script to see the system in action:

```bash
python demo.py
```

### Using the System

```python
from src.matrix_tracking.system import MatrixTrackingSystem

# Initialize the system
matrix_tracking = MatrixTrackingSystem()

# Plan a route
route_plan = matrix_tracking.plan_route(
    vehicle_id="truck001",
    start_lat=-23.5505,
    start_lng=-46.6333,  # São Paulo
    end_lat=-22.9068,
    end_lng=-43.1729,    # Rio de Janeiro
    departure_time="2025-08-25 08:00:00"
)

# Update vehicle location
matrix_tracking.update_vehicle_position(
    vehicle_id="truck001",
    latitude=-23.2193,
    longitude=-45.8889,
    timestamp="2025-08-25 10:30:00"
)

# Check for route anomalies
alerts = matrix_tracking.get_vehicle_alerts("truck001")
```

## Journey Duration Prediction

The system uses a Random Forest model trained on historical data to predict journey duration, considering:

- Distance between origin and destination
- Temporal patterns (time of day, day of week)
- Geographic characteristics
- Clusters of frequent locations

```python
from src.predict.duration_preditcor import DurationPredictor

predictor = DurationPredictor()
estimated_duration = predictor.predict(
    start_lng=-46.6333,
    start_lat=-23.5505,
    end_lng=-43.1729,
    end_lat=-22.9068,
    datetime="2025-08-25 08:00:00"
)
print(f"Estimated duration: {estimated_duration[0]/60:.1f} minutes")
```

## Development

### Feature Structure

The system uses the following feature categories for analysis and prediction:

- **Clustering**: Grouping of frequent origin and destination locations
- **Distance**: Distance calculations between geographic points
- **Geographical**: Geographic characteristics of locations
- **Interactions**: Interactions between different variables
- **Temporal**: Temporal characteristics (hour, day, month)

### Processing Pipeline

The feature pipeline transforms raw journey data into a feature set suitable for prediction models:

```python
from src.pipeline.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
processed_data = pipeline.fit(raw_data)
```

## Routing APIs Integration

### How Routing APIs Feed the Model

The routing engine provides the following data points that serve as inputs to our prediction models:

- **Optimized Route Path**: Series of waypoints representing the most efficient path
- **Distance Metrics**: Total distance and segment-by-segment breakdown
- **Estimated Duration**: Base travel time under ideal conditions
- **Traffic Conditions**: Real-time and historical traffic data (in production version)
- **Road Types**: Highway, urban, or rural classification of route segments

```python
# Example of how routing data feeds into the prediction pipeline
from src.matrix_tracking.routing_engine import routing_engine_calculate_route

# Get route data from the routing API
route_data = routing_engine_calculate_route(
    start_lat=-23.5505, 
    start_lng=-46.6333,
    end_lat=-22.9068, 
    end_lng=-43.1729,
    departure_time="2025-08-25 08:00:00"
)

# This data is then used by the prediction model to refine duration estimates
# and to detect anomalies during the journey
```

In a production environment, this would integrate with established providers like Google Maps API, HERE Maps, or OpenStreetMap Routing Service.

## Trajectory Database

The Matrix Cargo system maintains a comprehensive database of vehicle trajectories, which serves dual purposes: historical record-keeping and machine learning model training.

### Database Structure

The trajectory database stores:

- **Complete Trajectories**: Full GPS paths with timestamps
- **Journey Segments**: Individual segments for granular analysis
- **Performance Metrics**: Planned vs. actual duration, anomalies detected
- **Vehicle Metadata**: Vehicle types, cargo information, driver data

> For simulation and prototyping purposes, the trajectory database is implemented using in-memory pandas DataFrames instead of a persistent database. In a production environment, this would be replaced with a spatial database such as PostgreSQL with PostGIS or MongoDB with geospatial indexing support.

### How Trajectories Support the System

```python
# Example of how the trajectory database stores data
from src.matrix_tracking.trajectory_database import TrajectoryDatabase

# Initialize the database
db = TrajectoryDatabase()

# Store a completed journey
db.store_trajectory(
    vehicle_id="truck001",
    points=[(-23.5505, -46.6333), (-23.2193, -45.8889), (-22.9068, -43.1729)],
    timestamps=[
        pd.to_datetime("2025-08-25 08:00:00"),
        pd.to_datetime("2025-08-25 10:30:00"),
        pd.to_datetime("2025-08-25 13:15:00")
    ],
    metadata={
        "planned_duration": 18000,  # 5 hours in seconds
        "actual_duration": 19500,   # 5.4 hours in seconds
        "deviation": 0.083          # 8.3% longer than expected
    }
)
```

## AI for Predictive Decision-Making

### Delay Prediction

The system predicts potential delays by combining:
- Historical trajectory data
- Current vehicle position and progress
- Real-time traffic conditions (in production)

```python
# Example of how the system predicts potential delays
vehicle_status = matrix_tracking.get_vehicle_status("truck001")
if vehicle_status["delay_probability"] > 0.7:
    print(f"High risk of delay: {vehicle_status['estimated_delay']} minutes")
    print(f"Confidence: {vehicle_status['confidence_score']}")
```

### Anomaly Detection

The system uses machine learning to detect two types of anomalies:

1. **Route Anomalies**: When vehicles deviate significantly from planned routes
2. **Time Anomalies**: When journey durations differ significantly from predictions

```python
# Example of anomaly detection
anomalies = matrix_tracking.get_vehicle_anomalies("truck001")
for anomaly in anomalies:
    print(f"Detected {anomaly['type']} at {anomaly['timestamp']}")
    print(f"Details: {anomaly['details']}")
    
    if anomaly['type'] == 'route_deviation':
        print(f"Distance from expected path: {anomaly['deviation_distance']} km")
    elif anomaly['type'] == 'time_anomaly':
        print(f"Time deviation: {anomaly['deviation_percent']}%")
```
