import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

final apiClientProvider = Provider<ApiClient>((ref) {
  return ApiClient();
});

class ApiClient {
  static const String baseUrl = 'https://api.navswap.com/v1';
  
  late final Dio _dio;
  
  ApiClient() {
    _dio = Dio(
      BaseOptions(
        baseUrl: baseUrl,
        connectTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(seconds: 30),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );
    
    _dio.interceptors.add(
      PrettyDioLogger(
        requestHeader: true,
        requestBody: true,
        responseBody: true,
        responseHeader: false,
        error: true,
        compact: true,
      ),
    );
  }
  
  void setAuthToken(String token) {
    _dio.options.headers['Authorization'] = 'Bearer $token';
  }
  
  // Auth APIs
  Future<Map<String, dynamic>> login({
    required String emailOrPhone,
    required String password,
  }) async {
    try {
      final response = await _dio.post('/auth/login', data: {
        'emailOrPhone': emailOrPhone,
        'password': password,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> signup({
    required String name,
    required String email,
    required String phone,
    required String password,
  }) async {
    try {
      final response = await _dio.post('/auth/signup', data: {
        'name': name,
        'email': email,
        'phone': phone,
        'password': password,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> verifyOtp({
    required String phone,
    required String otp,
  }) async {
    try {
      final response = await _dio.post('/auth/verify-otp', data: {
        'phone': phone,
        'otp': otp,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> updateRole(String role) async {
    try {
      final response = await _dio.post('/auth/role', data: {
        'role': role,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  // Station APIs
  Future<List<dynamic>> getNearbyStations({
    required double lat,
    required double lng,
    int radius = 10,
  }) async {
    try {
      final response = await _dio.get('/stations/nearby', queryParameters: {
        'lat': lat,
        'lng': lng,
        'radius': radius,
      });
      final data = response.data as Map<String, dynamic>;
      return data['stations'] as List<dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> getStationDetail(String stationId) async {
    try {
      final response = await _dio.get('/stations/$stationId');
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> getAiRecommendation({
    required double lat,
    required double lng,
    int? batteryLevel,
  }) async {
    try {
      final response = await _dio.post('/ai/recommend-station', data: {
        'lat': lat,
        'lng': lng,
        'batteryLevel': batteryLevel,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  // Queue APIs
  Future<Map<String, dynamic>> joinQueue(String stationId) async {
    try {
      final response = await _dio.post('/queue/join', data: {
        'stationId': stationId,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> getQueueStatus(String queueId) async {
    try {
      final response = await _dio.get('/queue/$queueId');
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  // Task APIs (Transporter)
  Future<List<dynamic>> getAvailableTasks() async {
    try {
      final response = await _dio.get('/tasks/available');
      final data = response.data as Map<String, dynamic>;
      return data['tasks'] as List<dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> acceptTask(String taskId) async {
    try {
      final response = await _dio.post('/tasks/$taskId/accept');
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  Future<Map<String, dynamic>> updateTaskStatus({
    required String taskId,
    required String status,
  }) async {
    try {
      final response = await _dio.put('/tasks/$taskId/status', data: {
        'status': status,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  // AI Chat
  Future<Map<String, dynamic>> sendChatMessage(String message) async {
    try {
      final response = await _dio.post('/ai/chat', data: {
        'message': message,
      });
      return response.data as Map<String, dynamic>;
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }
  
  // Error handler
  String _handleError(DioException error) {
    switch (error.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        return 'Connection timeout. Please check your internet connection.';
      case DioExceptionType.badResponse:
        final statusCode = error.response?.statusCode;
        if (statusCode == 401) {
          return 'Unauthorized. Please login again.';
        } else if (statusCode == 404) {
          return 'Resource not found.';
        } else if (statusCode == 500) {
          return 'Server error. Please try again later.';
        }
        return 'Error: ${error.response?.data?['message'] ?? 'Something went wrong'}';
      case DioExceptionType.cancel:
        return 'Request cancelled.';
      default:
        return 'Network error. Please check your connection.';
    }
  }
}
