import 'package:dio/dio.dart';
import '../../../../core/constants/api_constants.dart';
import '../../../../core/exceptions/api_exceptions.dart';
import '../../../../core/services/dio_service.dart';
import '../../../../core/services/ai_inference_service.dart';
import '../models/station_model.dart';

abstract class CustomerRemoteDataSource {
  Future<RecommendationModel> getRecommendations({
    required String userId,
    required double latitude,
    required double longitude,
    String? vehicleType,
    int? batteryLevel,
  });

  Future<void> recordStationSelection({
    required String requestId,
    required String stationId,
  });

  Future<void> submitFeedback({
    required String requestId,
    required int rating,
  });

  Future<StationScoreModel> getStationScore(String stationId);
  Future<StationHealthModel> getStationHealth(String stationId);
  
  /// Get AI predictions for a station (on-device inference)
  Future<StationPredictions?> getAIPredictions(String stationId, Map<String, double> features);
}


class CustomerRemoteDataSourceImpl implements CustomerRemoteDataSource {
  final DioService dioService;

  CustomerRemoteDataSourceImpl(this.dioService);

  @override
  Future<RecommendationModel> getRecommendations({
    required String userId,
    required double latitude,
    required double longitude,
    String? vehicleType,
    int? batteryLevel,
  }) async {
    try {
      final response = await dioService.gateway.get(
        ApiConstants.recommendEndpoint,
        queryParameters: {
          'userId': userId,
          'lat': latitude,
          'lon': longitude,
          if (vehicleType != null) 'vehicleType': vehicleType,
          if (batteryLevel != null) 'batteryLevel': batteryLevel,
          'limit': 5,
        },
      );

      if (response.statusCode == 200) {
        final data = response.data['data'] ?? response.data;
        return RecommendationModel.fromJson(data);
      }
      // Fallback to mock data on non-200 response
      print('⚠️ API returned ${response.statusCode}, using mock data');
      return _getMockRecommendations(userId, latitude, longitude);
    } on DioException catch (e) {
      // Fallback to mock data on any network/server error
      print('⚠️ API error: ${e.message}, using mock recommendations');
      return _getMockRecommendations(userId, latitude, longitude);
    }
  }

  /// Generate mock recommendations when API is unavailable
  RecommendationModel _getMockRecommendations(
    String userId,
    double latitude,
    double longitude,
  ) {
    final now = DateTime.now();
    
    // Generate mock stations near the user's location
    final mockStations = [
      StationModel(
        stationId: 'mock_station_1',
        stationName: 'NavSwap Hub - Central',
        latitude: latitude + 0.005,
        longitude: longitude + 0.003,
        address: '123 Main Street, Central District',
        score: 92.5,
        rank: 1,
        estimatedWaitTime: 5,
        estimatedDistance: 0.8,
        availableChargers: 4,
        totalChargers: 6,
        chargerTypes: ['Type 2', 'CCS'],
        pricePerKwh: 12.5,
        status: 'operational',
        healthScore: 95,
      ),
      StationModel(
        stationId: 'mock_station_2',
        stationName: 'NavSwap Express - Mall',
        latitude: latitude - 0.008,
        longitude: longitude + 0.006,
        address: '456 Shopping Complex, Mall Road',
        score: 88.0,
        rank: 2,
        estimatedWaitTime: 8,
        estimatedDistance: 1.2,
        availableChargers: 3,
        totalChargers: 5,
        chargerTypes: ['Type 2', 'CHAdeMO'],
        pricePerKwh: 11.0,
        status: 'operational',
        healthScore: 90,
      ),
      StationModel(
        stationId: 'mock_station_3',
        stationName: 'NavSwap Point - Highway',
        latitude: latitude + 0.012,
        longitude: longitude - 0.009,
        address: '789 Highway Service Area',
        score: 85.5,
        rank: 3,
        estimatedWaitTime: 3,
        estimatedDistance: 2.5,
        availableChargers: 6,
        totalChargers: 8,
        chargerTypes: ['CCS', 'Tesla Supercharger'],
        pricePerKwh: 14.0,
        status: 'operational',
        healthScore: 88,
      ),
      StationModel(
        stationId: 'mock_station_4',
        stationName: 'NavSwap Green - Park',
        latitude: latitude - 0.003,
        longitude: longitude - 0.007,
        address: '321 Green Park Avenue',
        score: 82.0,
        rank: 4,
        estimatedWaitTime: 12,
        estimatedDistance: 1.8,
        availableChargers: 2,
        totalChargers: 4,
        chargerTypes: ['Type 2'],
        pricePerKwh: 10.0,
        status: 'operational',
        healthScore: 85,
      ),
      StationModel(
        stationId: 'mock_station_5',
        stationName: 'NavSwap Quick - Station',
        latitude: latitude + 0.009,
        longitude: longitude + 0.011,
        address: '555 Quick Charge Lane',
        score: 79.0,
        rank: 5,
        estimatedWaitTime: 15,
        estimatedDistance: 3.2,
        availableChargers: 5,
        totalChargers: 10,
        chargerTypes: ['Type 2', 'CCS', 'CHAdeMO'],
        pricePerKwh: 13.5,
        status: 'operational',
        healthScore: 92,
      ),
    ];

    return RecommendationModel(
      requestId: 'mock_${now.millisecondsSinceEpoch}',
      userId: userId,
      recommendations: mockStations,
      explanation: 'Showing nearby stations based on your location. (Offline mode - API unavailable)',
      generatedAt: now,
      expiresAt: now.add(const Duration(hours: 1)),
    );
  }


  @override
  Future<void> recordStationSelection({
    required String requestId,
    required String stationId,
  }) async {
    try {
      await dioService.recommendation.post(
        '${ApiConstants.recordSelectionEndpoint}/$requestId/select',
        data: {'stationId': stationId},
      );
    } on DioException catch (e) {
      throw _handleDioException(e);
    }
  }

  @override
  Future<void> submitFeedback({
    required String requestId,
    required int rating,
  }) async {
    try {
      await dioService.recommendation.post(
        '${ApiConstants.submitFeedbackEndpoint}/$requestId/feedback',
        data: {'rating': rating},
      );
    } on DioException catch (e) {
      throw _handleDioException(e);
    }
  }

  @override
  Future<StationScoreModel> getStationScore(String stationId) async {
    try {
      final response = await dioService.gateway.get(
        '${ApiConstants.stationScoreEndpoint}/$stationId/score',
      );

      if (response.statusCode == 200) {
        final data = response.data['data'] ?? response.data;
        return StationScoreModel.fromJson(data);
      }
      throw ServerException(
        message: 'Failed to get station score',
        originalException: response,
      );
    } on DioException catch (e) {
      throw _handleDioException(e);
    }
  }

  @override
  Future<StationHealthModel> getStationHealth(String stationId) async {
    try {
      final response = await dioService.gateway.get(
        '${ApiConstants.stationHealthEndpoint}/$stationId/health',
      );

      if (response.statusCode == 200) {
        final data = response.data['data'] ?? response.data;
        return StationHealthModel.fromJson(data);
      }
      throw ServerException(
        message: 'Failed to get station health',
        originalException: response,
      );
    } on DioException catch (e) {
      throw _handleDioException(e);
    }
  }

  ApiException _handleDioException(DioException e) {
    if (e.type == DioExceptionType.connectionTimeout ||
        e.type == DioExceptionType.receiveTimeout) {
      return NetworkException(
        message: 'Connection timeout. Please try again.',
        originalException: e,
      );
    }
    return ServerException(
      message: e.message ?? 'An error occurred',
      originalException: e,
    );
  }

  /// Get AI predictions using on-device ONNX models
  @override
  Future<StationPredictions?> getAIPredictions(
    String stationId, 
    Map<String, double> features,
  ) async {
    try {
      final aiService = AIInferenceService();
      await aiService.initialize();
      return await aiService.getStationPredictions(features);
    } catch (e) {
      print('⚠️ On-device AI prediction failed: $e');
      return null;
    }
  }
}
