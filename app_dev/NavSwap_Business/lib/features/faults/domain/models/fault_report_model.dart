class FaultReportModel {
  final String? message;
  final bool success;

  FaultReportModel({
    this.message,
    required this.success,
  });

  factory FaultReportModel.fromJson(Map<String, dynamic> json) {
    return FaultReportModel(
      message: json['message'] as String?,
      success: json['success'] as bool? ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'message': message,
      'success': success,
    };
  }
}
