import 'dart:convert';
import 'package:http/http.dart' as http;
import '../domain/models/station_score_model.dart';
import '../domain/models/station_health_model.dart';
import '../domain/models/station_predictions_model.dart';
import '../../../core/services/ai_inference_service.dart';

class StationApiService {
  static const String baseUrl =
      'http://ec2-52-89-235-59.us-west-2.compute.amazonaws.com:3000';

  // AI Inference Service instance
  final _aiService = AIInferenceService();

  Future<StationScoreModel?> getStationScore(String stationId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/station/$stationId/score'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return StationScoreModel.fromJson(data['data']);
      }
      return null;
    } catch (e) {
      print('Error fetching station score: $e');
      return null;
    }
  }

  Future<StationHealthModel?> getStationHealth(String stationId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/station/$stationId/health'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return StationHealthModel.fromJson(data['data']);
      }
      return null;
    } catch (e) {
      print('Error fetching station health: $e');
      return null;
    }
  }

  /// Get station predictions using on-device AI inference
  /// Falls back to API if local inference fails
  Future<StationPredictionsModel?> getStationPredictions(
      String stationId) async {
    // Try on-device inference first
    try {
      final predictions = await _getLocalPredictions(stationId);
      if (predictions != null) {
        print('✅ Using on-device AI predictions');
        return predictions;
      }
    } catch (e) {
      print('⚠️ On-device inference failed: $e');
    }

    // Fallback to API
    return _getApiPredictions(stationId);
  }

  /// Run predictions locally using ONNX models
  Future<StationPredictionsModel?> _getLocalPredictions(String stationId) async {
    await _aiService.initialize();

    // Create feature map from station data
    // In production, you'd get real station metrics here
    final features = _buildFeatureMap(stationId);

    final aiPredictions = await _aiService.getStationPredictions(features);

    final now = DateTime.now();
    final timestamp = now.millisecondsSinceEpoch;

    return StationPredictionsModel(
      loadForecast: LoadForecast(
        stationId: stationId,
        predictedLoad: aiPredictions.predictedQueueLength / 10.0, // Normalize to 0-1
        confidence: 0.85,
        peakTimeStart: '${now.hour + 2}:00',
        peakTimeEnd: '${now.hour + 4}:00',
        timestamp: timestamp,
      ),
      faultPrediction: FaultPrediction(
        stationId: stationId,
        faultProbability: aiPredictions.faultProbability,
        predictedFaultType: aiPredictions.faultRiskLevel == 1 ? 'charger_fault' : null,
        riskLevel: aiPredictions.faultRiskLabel.toLowerCase(),
        confidence: 0.90,
        timestamp: timestamp,
      ),
    );
  }

  /// Build feature map for AI inference
  /// TODO: Replace with real station metrics from your data source
  Map<String, double> _buildFeatureMap(String stationId) {
    // These are placeholder values - in production, fetch real metrics
    return {
      'queue_length': 3.0,
      'available_batteries': 15.0,
      'available_chargers': 8.0,
      'avg_wait_time': 12.0,
      'power_usage_kw': 150.0,
      'traffic_factor': 0.7,
      'station_reliability_score': 0.92,
      'energy_stability_index': 0.88,
      'hour_of_day': DateTime.now().hour.toDouble(),
      'day_of_week': DateTime.now().weekday.toDouble(),
      'is_peak_hour': (DateTime.now().hour >= 8 && DateTime.now().hour <= 10) ||
          (DateTime.now().hour >= 17 && DateTime.now().hour <= 19)
          ? 1.0
          : 0.0,
    };
  }

  /// Fallback: Get predictions from remote API
  Future<StationPredictionsModel?> _getApiPredictions(String stationId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/station/$stationId/predictions'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return StationPredictionsModel.fromJson(data['data']);
      }
      return null;
    } catch (e) {
      print('Error fetching station predictions from API: $e');
      return null;
    }
  }
}
