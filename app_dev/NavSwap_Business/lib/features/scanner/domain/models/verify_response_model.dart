class VerifyResponseModel {
  final String userId;
  final String userName;
  final String vehicleType;
  final String vehicleNumber;
  final int currentBatteryLevel;
  final String currentBatteryId;
  final String assignedBatteryId;
  final int assignedBatteryLevel;
  final String queuePosition;
  final String estimatedWaitTime;

  VerifyResponseModel({
    required this.userId,
    required this.userName,
    required this.vehicleType,
    required this.vehicleNumber,
    required this.currentBatteryLevel,
    required this.currentBatteryId,
    required this.assignedBatteryId,
    required this.assignedBatteryLevel,
    required this.queuePosition,
    required this.estimatedWaitTime,
  });

  factory VerifyResponseModel.fromJson(Map<String, dynamic> json) {
    return VerifyResponseModel(
      userId: json['userId'] ?? '',
      userName: json['userName'] ?? '',
      vehicleType: json['vehicleType'] ?? '',
      vehicleNumber: json['vehicleNumber'] ?? '',
      currentBatteryLevel: json['currentBatteryLevel'] ?? 0,
      currentBatteryId: json['currentBatteryId'] ?? '',
      assignedBatteryId: json['assignedBatteryId'] ?? '',
      assignedBatteryLevel: json['assignedBatteryLevel'] ?? 0,
      queuePosition: json['queuePosition'] ?? '',
      estimatedWaitTime: json['estimatedWaitTime'] ?? '',
    );
  }
}