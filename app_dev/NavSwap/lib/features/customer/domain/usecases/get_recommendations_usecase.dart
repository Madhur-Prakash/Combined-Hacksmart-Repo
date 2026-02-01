import '../entities/recommendation_entity.dart';
import '../../data/repositories/customer_repository.dart';

class GetRecommendationsUseCase {
  final CustomerRepository repository;

  GetRecommendationsUseCase(this.repository);

  Future<RecommendationEntity> call({
    required String userId,
    required double latitude,
    required double longitude,
    String? vehicleType,
    int? batteryLevel,
  }) {
    return repository.getRecommendations(
      userId: userId,
      latitude: latitude,
      longitude: longitude,
      vehicleType: vehicleType,
      batteryLevel: batteryLevel,
    );
  }
}

class RecordSelectionUseCase {
  final CustomerRepository repository;

  RecordSelectionUseCase(this.repository);

  Future<void> call({
    required String requestId,
    required String stationId,
  }) {
    return repository.recordStationSelection(
      requestId: requestId,
      stationId: stationId,
    );
  }
}

class SubmitFeedbackUseCase {
  final CustomerRepository repository;

  SubmitFeedbackUseCase(this.repository);

  Future<void> call({
    required String requestId,
    required int rating,
  }) {
    return repository.submitFeedback(
      requestId: requestId,
      rating: rating,
    );
  }
}

class GetStationScoreUseCase {
  final CustomerRepository repository;

  GetStationScoreUseCase(this.repository);

  Future<dynamic> call(String stationId) {
    return repository.getStationScore(stationId);
  }
}

class GetStationHealthUseCase {
  final CustomerRepository repository;

  GetStationHealthUseCase(this.repository);

  Future<dynamic> call(String stationId) {
    return repository.getStationHealth(stationId);
  }
}
