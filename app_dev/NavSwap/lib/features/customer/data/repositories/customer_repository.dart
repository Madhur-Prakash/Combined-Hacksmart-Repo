import '../../domain/entities/recommendation_entity.dart';
import '../datasources/customer_remote_datasource.dart';
import '../models/station_model.dart';

abstract class CustomerRepository {
  Future<RecommendationEntity> getRecommendations({
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
}

class CustomerRepositoryImpl implements CustomerRepository {
  final CustomerRemoteDataSource remoteDataSource;

  CustomerRepositoryImpl(this.remoteDataSource);

  @override
  Future<RecommendationEntity> getRecommendations({
    required String userId,
    required double latitude,
    required double longitude,
    String? vehicleType,
    int? batteryLevel,
  }) async {
    final model = await remoteDataSource.getRecommendations(
      userId: userId,
      latitude: latitude,
      longitude: longitude,
      vehicleType: vehicleType,
      batteryLevel: batteryLevel,
    );
    return model.toEntity();
  }

  @override
  Future<void> recordStationSelection({
    required String requestId,
    required String stationId,
  }) async {
    await remoteDataSource.recordStationSelection(
      requestId: requestId,
      stationId: stationId,
    );
  }

  @override
  Future<void> submitFeedback({
    required String requestId,
    required int rating,
  }) async {
    await remoteDataSource.submitFeedback(
      requestId: requestId,
      rating: rating,
    );
  }

  @override
  Future<StationScoreModel> getStationScore(String stationId) {
    return remoteDataSource.getStationScore(stationId);
  }

  @override
  Future<StationHealthModel> getStationHealth(String stationId) {
    return remoteDataSource.getStationHealth(stationId);
  }
}
