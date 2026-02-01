class RecommendationEntity {
  final String requestId;
  final String userId;
  final List<StationEntity> stations;
  final String explanation;
  final DateTime generatedAt;
  final DateTime expiresAt;

  RecommendationEntity({
    required this.requestId,
    required this.userId,
    required this.stations,
    required this.explanation,
    required this.generatedAt,
    required this.expiresAt,
  });
}

class StationEntity {
  final String stationId;
  final String stationName;
  final double latitude;
  final double longitude;
  final String address;
  final double score;
  final int rank;
  final int estimatedWaitTime;
  final double estimatedDistance;
  final int availableChargers;
  final int totalChargers;
  final List<String> chargerTypes;
  final double pricePerKwh;
  final String status;
  final int healthScore;

  StationEntity({
    required this.stationId,
    required this.stationName,
    required this.latitude,
    required this.longitude,
    required this.address,
    required this.score,
    required this.rank,
    required this.estimatedWaitTime,
    required this.estimatedDistance,
    required this.availableChargers,
    required this.totalChargers,
    required this.chargerTypes,
    required this.pricePerKwh,
    required this.status,
    required this.healthScore,
  });
}
