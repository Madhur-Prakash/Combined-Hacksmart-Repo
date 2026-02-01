class SwapResponseModel {
  final bool success;
  final String message;
  final String swapId;
  final DateTime timestamp;

  SwapResponseModel({
    required this.success,
    required this.message,
    required this.swapId,
    required this.timestamp,
  });

  factory SwapResponseModel.fromJson(Map<String, dynamic> json) {
    return SwapResponseModel(
      success: json['success'] ?? false,
      message: json['message'] ?? '',
      swapId: json['swapId'] ?? '',
      timestamp:
          DateTime.parse(json['timestamp'] ?? DateTime.now().toIso8601String()),
    );
  }
}
