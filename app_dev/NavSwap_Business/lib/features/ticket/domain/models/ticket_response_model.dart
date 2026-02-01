class TicketResponseModel {
  final String? message;
  final bool success;
  final String? ticketId;

  TicketResponseModel({
    this.message,
    required this.success,
    this.ticketId,
  });

  factory TicketResponseModel.fromJson(Map<String, dynamic> json) {
    return TicketResponseModel(
      message: json['message'] as String?,
      success: json['success'] as bool? ?? false,
      ticketId: json['ticketId'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'message': message,
      'success': success,
      'ticketId': ticketId,
    };
  }
}
