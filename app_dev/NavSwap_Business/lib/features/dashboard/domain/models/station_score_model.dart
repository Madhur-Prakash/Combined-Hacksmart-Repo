class StationScoreModel {
  final String stationId;
  final double overallScore;
  final ComponentScores componentScores;
  final int rank;
  final double confidence;
  final int timestamp;

  StationScoreModel({
    required this.stationId,
    required this.overallScore,
    required this.componentScores,
    required this.rank,
    required this.confidence,
    required this.timestamp,
  });

  factory StationScoreModel.fromJson(Map<String, dynamic> json) {
    return StationScoreModel(
      stationId: json['stationId'],
      overallScore: json['overallScore'],
      componentScores: ComponentScores.fromJson(json['componentScores']),
      rank: json['rank'],
      confidence: json['confidence'],
      timestamp: json['timestamp'],
    );
  }
}

class ComponentScores {
  final double waitTimeScore;
  final double availabilityScore;
  final double reliabilityScore;
  final double distanceScore;
  final double energyStabilityScore;

  ComponentScores({
    required this.waitTimeScore,
    required this.availabilityScore,
    required this.reliabilityScore,
    required this.distanceScore,
    required this.energyStabilityScore,
  });

  factory ComponentScores.fromJson(Map<String, dynamic> json) {
    return ComponentScores(
      waitTimeScore: json['waitTimeScore'],
      availabilityScore: json['availabilityScore'],
      reliabilityScore: json['reliabilityScore'],
      distanceScore: json['distanceScore'],
      energyStabilityScore: json['energyStabilityScore'],
    );
  }
}