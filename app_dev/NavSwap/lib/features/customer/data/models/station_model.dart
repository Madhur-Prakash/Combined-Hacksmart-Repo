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

  factory StationModel.fromJson(Map<String, dynamic> json) {
    return StationModel(
      stationId: json['stationId'] ?? '',
      stationName: json['stationName'] ?? 'Unknown Station',
      latitude: (json['location']?['latitude'] ?? 0.0).toDouble(),
      longitude: (json['location']?['longitude'] ?? 0.0).toDouble(),
      address: json['address'] ?? '',
      score: ((json['score'] as num?) ?? 0).toDouble(),
      rank: (json['rank'] as num?)?.toInt() ?? 0,
      estimatedWaitTime: (json['estimatedWaitTime'] as num?)?.toInt() ?? 0,
      estimatedDistance: ((json['estimatedDistance'] as num?) ?? 0).toDouble(),
      availableChargers: (json['availableChargers'] as num?)?.toInt() ?? 0,
      totalChargers: (json['totalChargers'] as num?)?.toInt() ?? 0,
      chargerTypes: List<String>.from(json['chargerTypes'] ?? []),
      pricePerKwh: ((json['pricePerKwh'] as num?) ?? 0).toDouble(),
      status: json['status'] ?? 'operational',
      healthScore: (json['healthScore'] as num?)?.toInt() ?? 100,
    );
  }

  Map<String, dynamic> toJson() => {
        'stationId': stationId,
        'stationName': stationName,
        'location': {
          'latitude': latitude,
          'longitude': longitude,
        },
        'address': address,
        'score': score,
        'rank': rank,
        'estimatedWaitTime': estimatedWaitTime,
        'estimatedDistance': estimatedDistance,
        'availableChargers': availableChargers,
        'totalChargers': totalChargers,
        'chargerTypes': chargerTypes,
        'pricePerKwh': pricePerKwh,
        'status': status,
        'healthScore': healthScore,
      };

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

class RecommendationModel {
  final String requestId;
  final String userId;
  final List<StationModel> recommendations;
  final String explanation;
  final DateTime generatedAt;
  final DateTime expiresAt;

  RecommendationModel({
    required this.requestId,
    required this.userId,
    required this.recommendations,
    required this.explanation,
    required this.generatedAt,
    required this.expiresAt,
  });

  factory RecommendationModel.fromJson(Map<String, dynamic> json) {
    return RecommendationModel(
      requestId: json['requestId'] ?? '',
      userId: json['userId'] ?? '',
      recommendations: (json['recommendations'] as List<dynamic>?)
              ?.map((e) => StationModel.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      explanation: json['explanation'] ?? '',
      generatedAt: json['generatedAt'] != null
          ? DateTime.parse(json['generatedAt'])
          : DateTime.now(),
      expiresAt: json['expiresAt'] != null
          ? DateTime.parse(json['expiresAt'])
          : DateTime.now().add(const Duration(hours: 1)),
    );
  }

  Map<String, dynamic> toJson() => {
        'requestId': requestId,
        'userId': userId,
        'recommendations': recommendations.map((e) => e.toJson()).toList(),
        'explanation': explanation,
        'generatedAt': generatedAt.toIso8601String(),
        'expiresAt': expiresAt.toIso8601String(),
      };
}

extension RecommendationModelToEntity on RecommendationModel {
  RecommendationEntity toEntity() {
    return RecommendationEntity(
      requestId: requestId,
      userId: userId,
      stations: recommendations
          .map((e) => StationModel(
                stationId: e.stationId,
                stationName: e.stationName,
                latitude: e.latitude,
                longitude: e.longitude,
                address: e.address,
                score: e.score,
                rank: e.rank,
                estimatedWaitTime: e.estimatedWaitTime,
                estimatedDistance: e.estimatedDistance,
                availableChargers: e.availableChargers,
                totalChargers: e.totalChargers,
                chargerTypes: e.chargerTypes,
                pricePerKwh: e.pricePerKwh,
                status: e.status,
                healthScore: e.healthScore,
              ).toEntity())
          .toList(),
      explanation: explanation,
      generatedAt: generatedAt,
      expiresAt: expiresAt,
    );
  }
}

class StationScoreModel {
  final String stationId;
  final double overallScore;
  final double waitTimeScore;
  final double reliabilityScore;
  final double energyStabilityScore;
  final double chargerAvailabilityScore;

  StationScoreModel({
    required this.stationId,
    required this.overallScore,
    required this.waitTimeScore,
    required this.reliabilityScore,
    required this.energyStabilityScore,
    required this.chargerAvailabilityScore,
  });

  factory StationScoreModel.fromJson(Map<String, dynamic> json) {
    return StationScoreModel(
      stationId: json['stationId'] ?? '',
      overallScore: ((json['overallScore'] as num?) ?? 0).toDouble(),
      waitTimeScore: ((json['waitTimeScore'] as num?) ?? 0).toDouble(),
      reliabilityScore: ((json['reliabilityScore'] as num?) ?? 0).toDouble(),
      energyStabilityScore:
          ((json['energyStabilityScore'] as num?) ?? 0).toDouble(),
      chargerAvailabilityScore:
          ((json['chargerAvailabilityScore'] as num?) ?? 0).toDouble(),
    );
  }
}

class StationHealthModel {
  final String stationId;
  final String status;
  final int healthScore;
  final List<String> activeAlerts;

  StationHealthModel({
    required this.stationId,
    required this.status,
    required this.healthScore,
    required this.activeAlerts,
  });

  factory StationHealthModel.fromJson(Map<String, dynamic> json) {
    return StationHealthModel(
      stationId: json['stationId'] ?? '',
      status: json['status'] ?? 'unknown',
      healthScore: (json['healthScore'] as num?)?.toInt() ?? 0,
      activeAlerts: List<String>.from(json['activeAlerts'] ?? []),
    );
  }
}
