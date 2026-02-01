class TaskModel {
  final String id;
  final String pickupStationId;
  final String pickupStationName;
  final String dropStationId;
  final String dropStationName;
  final double distance;
  final int batteryCount;
  final double estimatedCredits;
  final String status;
  final DateTime? acceptedAt;
  final DateTime? completedAt;
  final String? proofImageUrl;

  TaskModel({
    required this.id,
    required this.pickupStationId,
    required this.pickupStationName,
    required this.dropStationId,
    required this.dropStationName,
    required this.distance,
    required this.batteryCount,
    required this.estimatedCredits,
    required this.status,
    this.acceptedAt,
    this.completedAt,
    this.proofImageUrl,
  });

  factory TaskModel.fromJson(Map<String, dynamic> json) {
    return TaskModel(
      id: json['id'] as String,
      pickupStationId: json['pickupStationId'] as String,
      pickupStationName: json['pickupStationName'] as String,
      dropStationId: json['dropStationId'] as String,
      dropStationName: json['dropStationName'] as String,
      distance: (json['distance'] as num).toDouble(),
      batteryCount: json['batteryCount'] as int,
      estimatedCredits: (json['estimatedCredits'] as num).toDouble(),
      status: json['status'] as String,
      acceptedAt: json['acceptedAt'] != null
          ? DateTime.parse(json['acceptedAt'] as String)
          : null,
      completedAt: json['completedAt'] != null
          ? DateTime.parse(json['completedAt'] as String)
          : null,
      proofImageUrl: json['proofImageUrl'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'pickupStationId': pickupStationId,
      'pickupStationName': pickupStationName,
      'dropStationId': dropStationId,
      'dropStationName': dropStationName,
      'distance': distance,
      'batteryCount': batteryCount,
      'estimatedCredits': estimatedCredits,
      'status': status,
      'acceptedAt': acceptedAt?.toIso8601String(),
      'completedAt': completedAt?.toIso8601String(),
      'proofImageUrl': proofImageUrl,
    };
  }
}

class TransporterStatsModel {
  final double totalEarnings;
  final int totalDeliveries;
  final double efficiencyScore;
  final double onTimePercentage;
  final int todayDeliveries;
  final double todayEarnings;
  final String tierBadge;

  TransporterStatsModel({
    required this.totalEarnings,
    required this.totalDeliveries,
    required this.efficiencyScore,
    required this.onTimePercentage,
    required this.todayDeliveries,
    required this.todayEarnings,
    required this.tierBadge,
  });

  factory TransporterStatsModel.fromJson(Map<String, dynamic> json) {
    return TransporterStatsModel(
      totalEarnings: (json['totalEarnings'] as num).toDouble(),
      totalDeliveries: json['totalDeliveries'] as int,
      efficiencyScore: (json['efficiencyScore'] as num).toDouble(),
      onTimePercentage: (json['onTimePercentage'] as num).toDouble(),
      todayDeliveries: json['todayDeliveries'] as int,
      todayEarnings: (json['todayEarnings'] as num).toDouble(),
      tierBadge: json['tierBadge'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'totalEarnings': totalEarnings,
      'totalDeliveries': totalDeliveries,
      'efficiencyScore': efficiencyScore,
      'onTimePercentage': onTimePercentage,
      'todayDeliveries': todayDeliveries,
      'todayEarnings': todayEarnings,
      'tierBadge': tierBadge,
    };
  }
}
