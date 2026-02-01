import 'package:navswap_app/features/customer/domain/entities/recommendation_entity.dart';

class StationModel {
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

  StationModel({
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

  StationEntity toEntity() {
    return StationEntity(
      stationId: stationId,
      stationName: stationName,
      latitude: latitude,
      longitude: longitude,
      address: address,
      score: score,
      rank: rank,
      estimatedWaitTime: estimatedWaitTime,
      estimatedDistance: estimatedDistance,
      availableChargers: availableChargers,
      totalChargers: totalChargers,
      chargerTypes: chargerTypes,
      pricePerKwh: pricePerKwh,
      status: status,
      healthScore: healthScore,
    );
  }
}
