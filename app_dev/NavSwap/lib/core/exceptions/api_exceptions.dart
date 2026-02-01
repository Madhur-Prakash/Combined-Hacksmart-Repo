class ApiException implements Exception {
  final String message;
  final String? code;
  final dynamic originalException;

  ApiException({
    required this.message,
    this.code,
    this.originalException,
  });

  @override
  String toString() => message;
}

class NetworkException extends ApiException {
  NetworkException({
    required String message,
    dynamic originalException,
  }) : super(
          message: message,
          code: 'NETWORK_ERROR',
          originalException: originalException,
        );
}

class ServerException extends ApiException {
  ServerException({
    required String message,
    dynamic originalException,
  }) : super(
          message: message,
          code: 'SERVER_ERROR',
          originalException: originalException,
        );
}
