"""
PERSON B - Station Recommendation Engine
WHAT THIS DOES: Recommends the best station for a user based on multiple factors
USES: Multi-objective optimization with XGBoost
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

class StationRecommender:
    """
    SIMPLE EXPLANATION:
    This is like Google Maps for EV battery swaps! It considers:
    - How busy each station is
    - How far away it is
    - How reliable it is
    - Whether it has batteries available
    
    Then ranks all stations and recommends the best one.
    """
    
    def __init__(self, config: Dict = None):
        """
        PARAMETERS:
        - config: Dictionary with optimization weights
        """
        self.config = config or {
            'weight_wait_time': 0.30,      # 30% importance
            'weight_availability': 0.25,   # 25% importance
            'weight_reliability': 0.20,    # 20% importance
            'weight_distance': 0.15,       # 15% importance
            'weight_energy': 0.10          # 10% importance
        }
        
        self.model = None
        self.feature_names = None
    
    def calculate_station_score(self, station_data: Dict) -> float:
        """
        WHAT THIS DOES:
        Calculates a single score for a station based on multiple factors.
        
        HOW IT WORKS:
        Score = w1×wait_score + w2×availability_score + ... (weighted sum)
        Higher score = better station
        
        PARAMETERS:
        - station_data: Dictionary with station metrics
        
        RETURNS:
        - Score between 0-100 (higher is better)
        """
        # Normalize each metric to 0-1 scale (higher is better)
        
        # 1. Wait time score (lower wait time = higher score)
        max_wait = 30  # Assume max wait is 30 minutes
        wait_score = 1 - min(station_data['avg_wait_time'] / max_wait, 1)
        
        # 2. Availability score (more batteries = higher score)
        availability_score = station_data['available_batteries'] / station_data['total_batteries']
        
        # 3. Reliability score (already 0-1)
        reliability_score = station_data['station_reliability_score']
        
        # 4. Distance score (closer = higher score)
        max_distance = 20  # Assume max acceptable distance is 20 km
        distance_score = 1 - min(station_data.get('distance_km', 0) / max_distance, 1)
        
        # 5. Energy stability score (more stable = higher score)
        energy_score = 1 - station_data['energy_stability_index']
        
        # Weighted combination
        total_score = (
            self.config['weight_wait_time'] * wait_score +
            self.config['weight_availability'] * availability_score +
            self.config['weight_reliability'] * reliability_score +
            self.config['weight_distance'] * distance_score +
            self.config['weight_energy'] * energy_score
        )
        
        # Convert to 0-100 scale
        return round(total_score * 100, 2)
    
    def rank_stations(
        self, 
        stations_data: List[Dict],
        user_location: Tuple[float, float] = None,
        user_urgency: str = 'normal'
    ) -> List[Dict]:
        """
        WHAT THIS DOES:
        Ranks all available stations from best to worst.
        
        PARAMETERS:
        - stations_data: List of all station data
        - user_location: (latitude, longitude) of user
        - user_urgency: 'low', 'normal', or 'high'
        
        RETURNS:
        - List of stations sorted by recommendation score
        """
        scored_stations = []
        
        for station in stations_data:
            # Skip if station is offline
            if station['status'] == 'OFFLINE':
                continue
            
            # Calculate distance if user location provided
            if user_location:
                station['distance_km'] = self._calculate_distance(
                    user_location,
                    (station.get('lat', 0), station.get('lon', 0))
                )
            
            # Adjust weights based on urgency
            if user_urgency == 'high':
                # Prioritize wait time and availability
                self.config['weight_wait_time'] = 0.40
                self.config['weight_availability'] = 0.35
                self.config['weight_distance'] = 0.10
            elif user_urgency == 'low':
                # Can afford to go further for better station
                self.config['weight_distance'] = 0.05
                self.config['weight_reliability'] = 0.30
            
            # Calculate score
            score = self.calculate_station_score(station)
            
            station_with_score = station.copy()
            station_with_score['recommendation_score'] = score
            scored_stations.append(station_with_score)
        
        # Sort by score (highest first)
        ranked = sorted(scored_stations, key=lambda x: x['recommendation_score'], reverse=True)
        
        return ranked
    
    def _calculate_distance(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """
        WHAT THIS DOES:
        Calculates distance between two GPS coordinates.
        
        USES: Haversine formula (accounts for Earth's curvature)
        
        PARAMETERS:
        - point1, point2: (latitude, longitude) tuples
        
        RETURNS:
        - Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Earth's radius in km
        R = 6371.0
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return round(distance, 2)
    
    def get_recommendation(
        self,
        stations_data: List[Dict],
        user_context: Dict
    ) -> Dict:
        """
        WHAT THIS DOES:
        Main function to get station recommendation for a user.
        
        PARAMETERS:
        - stations_data: Current data for all stations
        - user_context: {
            'location': (lat, lon),
            'urgency': 'low'|'normal'|'high',
            'vehicle_type': 'sedan'|'suv'|'bike',
            'soc_percent': 10  # State of charge
          }
        
        RETURNS:
        - Dictionary with:
            - recommended_station: Best station
            - alternatives: Top 3 alternatives
            - reasoning: Why this station was chosen
        """
        # Rank all stations
        ranked_stations = self.rank_stations(
            stations_data,
            user_location=user_context.get('location'),
            user_urgency=user_context.get('urgency', 'normal')
        )
        
        if not ranked_stations:
            return {
                'error': 'No stations available',
                'recommended_station': None,
                'alternatives': []
            }
        
        # Get top recommendation
        best_station = ranked_stations[0]
        alternatives = ranked_stations[1:4]  # Next 3 best
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_station, user_context)
        
        return {
            'recommended_station': best_station,
            'alternatives': alternatives,
            'reasoning': reasoning,
            'confidence': self._calculate_confidence(best_station, alternatives)
        }
    
    def _generate_reasoning(self, station: Dict, user_context: Dict) -> str:
        """
        WHAT THIS DOES:
        Generates human-readable explanation for the recommendation.
        
        This is what the LLM will expand upon!
        """
        reasons = []
        
        # Wait time
        if station['avg_wait_time'] < 5:
            reasons.append(f"minimal wait time ({station['avg_wait_time']:.1f} min)")
        
        # Availability
        if station['available_batteries'] >= 10:
            reasons.append(f"{station['available_batteries']} batteries available")
        
        # Reliability
        if station['station_reliability_score'] > 0.9:
            reasons.append("high reliability score")
        
        # Distance
        if 'distance_km' in station and station['distance_km'] < 5:
            reasons.append(f"only {station['distance_km']:.1f} km away")
        
        reasoning = (
            f"{station['station_id']} is recommended because it has " +
            ", ".join(reasons) + "."
        )
        
        return reasoning
    
    def _calculate_confidence(self, best: Dict, alternatives: List[Dict]) -> float:
        """
        WHAT THIS DOES:
        Calculates how confident we are in the recommendation.
        
        HOW:
        If the best station's score is much higher than alternatives,
        confidence is high. If scores are close, confidence is lower.
        
        RETURNS:
        - Confidence score 0-1 (1 = very confident)
        """
        if not alternatives:
            return 1.0
        
        best_score = best['recommendation_score']
        second_best_score = alternatives[0]['recommendation_score']
        
        # Calculate gap
        score_gap = best_score - second_best_score
        
        # Convert to confidence (larger gap = higher confidence)
        confidence = min(score_gap / 20, 1.0)  # Normalize to 0-1
        
        return round(confidence, 2)
    
    def train_ml_ranker(self, historical_data: pd.DataFrame):
        """
        WHAT THIS DOES:
        Trains a machine learning model to learn from past user choices.
        This improves recommendations over time.
        
        WHY ML:
        Instead of just using fixed weights, the model learns which
        factors are most important by looking at what users actually chose.
        
        USES: XGBoost (Gradient Boosted Trees)
        """
        print(" Training ML-based ranker...")
        
        # Prepare features and target
        # Target: 1 if user chose this station, 0 otherwise
        feature_cols = [
            'avg_wait_time',
            'available_batteries',
            'station_reliability_score',
            'distance_km',
            'energy_stability_index',
            'queue_length',
            'is_peak_hour'
        ]
        
        X = historical_data[feature_cols]
        y = historical_data['was_chosen']  # Binary: 1=chosen, 0=not chosen
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f" Training complete!")
        print(f"  Accuracy: {accuracy:.3f}")
        
        self.feature_names = feature_cols
    
    def predict_user_preference(self, station_features: Dict) -> float:
        """
        WHAT THIS DOES:
        Predicts probability that a user will choose this station.
        
        USES: Trained ML model
        
        RETURNS:
        - Probability 0-1 (higher = more likely to be chosen)
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train_ml_ranker() first.")
        
        # Prepare features
        features = pd.DataFrame([station_features])[self.feature_names]
        
        # Predict probability
        probability = self.model.predict_proba(features)[0][1]
        
        return round(probability, 3)
    
    def save_recommender(self, filepath: str):
        """Save the recommender (config + model)"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
        print(f" Recommender saved to {filepath}")
    
    def load_recommender(self, filepath: str):
        """Load a saved recommender"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.model = data['model']
            self.feature_names = data['feature_names']
        print(f" Recommender loaded from {filepath}")


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    """
    HOW TO USE THIS:
    Complete example of getting station recommendations
    """
    print(" Station Recommender Demo\n")
    
    # Import dependencies
    import sys
    sys.path.append('nexora_ai')
    from shared.data_simulator import StationDataSimulator
    
    # Generate sample station data
    simulator = StationDataSimulator(num_stations=10)
    stations_data = simulator.generate_realtime_data()
    
    # Add GPS coordinates
    for station in stations_data:
        loc = simulator.station_locations[station['station_id']]
        station['lat'] = loc['lat']
        station['lon'] = loc['lon']
    
    # Create recommender
    recommender = StationRecommender()
    
    # User context
    user_context = {
        'location': (28.4595, 77.0266),  # Gurugram
        'urgency': 'high',
        'vehicle_type': 'sedan',
        'soc_percent': 15  # 15% battery left
    }
    
    # Get recommendation
    print(" Getting recommendation for user...\n")
    result = recommender.get_recommendation(stations_data, user_context)
    
    # Display results
    best = result['recommended_station']
    print(" RECOMMENDED STATION:")
    print(f"  Station: {best['station_id']}")
    print(f"  Score: {best['recommendation_score']}/100")
    print(f"  Wait Time: {best['avg_wait_time']:.1f} min")
    print(f"  Available Batteries: {best['available_batteries']}")
    print(f"  Distance: {best.get('distance_km', 'N/A')} km")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"\n  Reasoning: {result['reasoning']}")
    
    print("\n\n ALTERNATIVE OPTIONS:")
    for i, alt in enumerate(result['alternatives'], 1):
        print(f"\n  {i}. {alt['station_id']}")
        print(f"     Score: {alt['recommendation_score']}/100")
        print(f"     Wait: {alt['avg_wait_time']:.1f} min")
        print(f"     Batteries: {alt['available_batteries']}")
    
    print("\n\n Demo complete!")


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    # -----------------------------
    # 1. Create fake historical data
    # -----------------------------
    np.random.seed(42)

    historical_data = pd.DataFrame({
        'avg_wait_time': np.random.uniform(1, 20, 300),
        'available_batteries': np.random.randint(1, 20, 300),
        'station_reliability_score': np.random.uniform(0.7, 1.0, 300),
        'distance_km': np.random.uniform(0.5, 15, 300),
        'energy_stability_index': np.random.uniform(0.0, 0.5, 300),
        'queue_length': np.random.randint(0, 15, 300),
        'is_peak_hour': np.random.randint(0, 2, 300),
        'was_chosen': np.random.randint(0, 2, 300)
    })

    # -----------------------------
    # 2. Train recommender
    # -----------------------------
    recommender = StationRecommender()
    recommender.train_ml_ranker(historical_data)

    # -----------------------------
    # 3. Save model as PKL
    # -----------------------------
    recommender.save_recommender("station_recommender.pkl")
