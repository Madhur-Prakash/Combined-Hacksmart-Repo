/// AI Inference Service for on-device predictions
///
/// This service provides AI predictions using mock data for development.
/// In production, this would load ONNX models from assets.
library;

import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';

/// Singleton service for AI model inference
class AIInferenceService {
  static final AIInferenceService _instance = AIInferenceService._internal();
  factory AIInferenceService() => _instance;
  AIInferenceService._internal();

  // Preprocessing parameters
  Map<String, dynamic>? _scalerParams;
  List<String>? _actionLabels;

  bool _isInitialized = false;
  final _random = Random();

  /// Initialize the service and load preprocessing parameters
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load preprocessing parameters
      await _loadScalerParams();
      await _loadLabelEncoder();

      _isInitialized = true;
      print('✅ AI Inference Service initialized (Mock Mode)');
    } catch (e) {
      print('❌ Failed to initialize AI service: $e');
      // Continue with defaults
      _actionLabels = ['NORMAL', 'MAINTENANCE', 'URGENT', 'SHUTDOWN'];
      _isInitialized = true;
    }
  }

  Future<void> _loadScalerParams() async {
    try {
      final jsonStr =
          await rootBundle.loadString('assets/models/scaler_params.json');
      _scalerParams = json.decode(jsonStr);
    } catch (e) {
      print('⚠️ Could not load scaler params: $e');
    }
  }

  Future<void> _loadLabelEncoder() async {
    try {
      final jsonStr =
          await rootBundle.loadString('assets/models/label_encoder.json');
      final data = json.decode(jsonStr);
      _actionLabels = List<String>.from(data['classes']);
    } catch (e) {
      print('⚠️ Could not load label encoder: $e');
    }
  }

  /// Preprocess input features using loaded scaler parameters
  List<double> _preprocess(Map<String, double> features) {
    if (_scalerParams == null) {
      return features.values.toList();
    }

    final mean = List<double>.from(_scalerParams!['mean']);
    final scale = List<double>.from(_scalerParams!['scale']);
    final featureNames = _scalerParams!['feature_names'] as List<dynamic>?;

    List<double> result = [];

    if (featureNames != null) {
      for (int i = 0; i < featureNames.length; i++) {
        final name = featureNames[i] as String;
        final value = features[name] ?? 0.0;
        result.add((value - mean[i]) / scale[i]);
      }
    } else {
      int i = 0;
      for (final value in features.values) {
        if (i < mean.length) {
          result.add((value - mean[i]) / scale[i]);
        } else {
          result.add(value);
        }
        i++;
      }
    }

    return result;
  }

  /// Mock inference for development
  List<double> _mockInference(String modelType, List<double> input) {
    switch (modelType) {
      case 'queue':
        return [_random.nextDouble() * 10]; // 0-10 queue length
      case 'wait':
        return [_random.nextDouble() * 30]; // 0-30 minutes wait
      case 'fault':
        final prob = _random.nextDouble();
        return [prob < 0.8 ? 0 : 1, prob]; // Low fault probability
      case 'action':
        final probs = List.generate(4, (_) => _random.nextDouble());
        final sum = probs.reduce((a, b) => a + b);
        return probs.map((p) => p / sum).toList(); // Normalized probabilities
      default:
        return [0.0];
    }
  }

  /// Predict queue length for a station
  Future<double> predictQueueLength(Map<String, double> features) async {
    if (!_isInitialized) await initialize();

    final input = _preprocess(features);
    final output = _mockInference('queue', input);
    return output.first;
  }

  /// Predict wait time for a station
  Future<double> predictWaitTime(Map<String, double> features) async {
    if (!_isInitialized) await initialize();

    final input = _preprocess(features);
    final output = _mockInference('wait', input);
    return output.first;
  }

  /// Predict fault risk for a station
  Future<({int riskLevel, double probability})> predictFaultRisk(
    Map<String, double> features,
  ) async {
    if (!_isInitialized) await initialize();

    final input = _preprocess(features);
    final output = _mockInference('fault', input);

    if (output.length >= 2) {
      final prob = output[1];
      return (riskLevel: prob > 0.5 ? 1 : 0, probability: prob);
    } else {
      return (riskLevel: output.first.round(), probability: output.first);
    }
  }

  /// Predict recommended action for a station
  Future<({String action, Map<String, double> probabilities})> predictAction(
    Map<String, double> features,
  ) async {
    if (!_isInitialized) await initialize();

    final input = _preprocess(features);
    final output = _mockInference('action', input);

    int predictedClass = 0;
    double maxProb = 0;
    Map<String, double> probabilities = {};

    final labels = _actionLabels ?? ['NORMAL', 'MAINTENANCE', 'URGENT', 'SHUTDOWN'];

    if (output.length >= labels.length) {
      for (int i = 0; i < labels.length; i++) {
        probabilities[labels[i]] = output[i];
        if (output[i] > maxProb) {
          maxProb = output[i];
          predictedClass = i;
        }
      }
    } else {
      predictedClass = output.first.round();
      probabilities = {'predicted': output.first};
    }

    final actionLabel = predictedClass < labels.length
        ? labels[predictedClass]
        : 'NORMAL';

    return (action: actionLabel, probabilities: probabilities);
  }

  /// Get all predictions for a station at once
  Future<StationPredictions> getStationPredictions(
    Map<String, double> features,
  ) async {
    final queue = await predictQueueLength(features);
    final wait = await predictWaitTime(features);
    final fault = await predictFaultRisk(features);
    final action = await predictAction(features);

    return StationPredictions(
      predictedQueueLength: queue,
      predictedWaitTime: wait,
      faultRiskLevel: fault.riskLevel,
      faultProbability: fault.probability,
      recommendedAction: action.action,
      actionProbabilities: action.probabilities,
    );
  }

  /// Dispose of resources
  void dispose() {
    _isInitialized = false;
  }
}

/// Data class for station predictions
class StationPredictions {
  final double predictedQueueLength;
  final double predictedWaitTime;
  final int faultRiskLevel;
  final double faultProbability;
  final String recommendedAction;
  final Map<String, double> actionProbabilities;

  StationPredictions({
    required this.predictedQueueLength,
    required this.predictedWaitTime,
    required this.faultRiskLevel,
    required this.faultProbability,
    required this.recommendedAction,
    required this.actionProbabilities,
  });

  String get faultRiskLabel => faultRiskLevel == 1 ? 'HIGH' : 'LOW';
}

/// Global instance for easy access
final aiInferenceService = AIInferenceService();
