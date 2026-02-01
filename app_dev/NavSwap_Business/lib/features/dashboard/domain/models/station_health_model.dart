class StationHealthModel {
  final String stationId;
  final String status;
  final String lastMaintenanceDate;
  final double uptimePercentage;
  final List<dynamic> activeAlerts;
  final int healthScore;
  final int timestamp;

  StationHealthModel({
    required this.stationId,
    required this.status,
    required this.lastMaintenanceDate,
    required this.uptimePercentage,
    required this.activeAlerts,
    required this.healthScore,
    required this.timestamp,
  });

  factory StationHealthModel.fromJson(Map<String, dynamic> json) {
    return StationHealthModel(
      stationId: json['stationId'],
      status: json['status'],
      lastMaintenanceDate: json['lastMaintenanceDate'],
      uptimePercentage: json['uptimePercentage'],
      activeAlerts: json['activeAlerts'] ?? [],
      healthScore: json['healthScore'],
      timestamp: json['timestamp'],
    );
  }
}
