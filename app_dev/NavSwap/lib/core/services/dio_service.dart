import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';
import '../constants/api_constants.dart';

class DioService {
  static final DioService _instance = DioService._internal();

  late Dio _gatewayDio;
  late Dio _recommendationDio;
  late Dio _ingestionDio;

  DioService._internal() {
    _initializeDio();
  }

  factory DioService() {
    return _instance;
  }

  void _initializeDio() {
    // Gateway Dio
    _gatewayDio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.apiGatewayBaseUrl,
        connectTimeout: ApiConstants.connectionTimeout,
        receiveTimeout: ApiConstants.receiveTimeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    // Recommendation Service Dio
    _recommendationDio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.recommendationServiceUrl,
        connectTimeout: ApiConstants.connectionTimeout,
        receiveTimeout: ApiConstants.receiveTimeout,
        headers: {
          'Content-Type': 'application/json',
        },
      ),
    );

    // Ingestion Service Dio
    _ingestionDio = Dio(
      BaseOptions(
        baseUrl: ApiConstants.ingestionServiceUrl,
        connectTimeout: ApiConstants.connectionTimeout,
        receiveTimeout: ApiConstants.receiveTimeout,
        headers: {
          'Content-Type': 'application/json',
        },
      ),
    );

    if (kDebugMode) {
      _gatewayDio.interceptors.add(
        PrettyDioLogger(
          requestHeader: true,
          requestBody: true,
          responseBody: true,
          responseHeader: false,
          compact: true,
        ),
      );

      _recommendationDio.interceptors.add(
        PrettyDioLogger(
          requestHeader: true,
          requestBody: true,
          responseBody: true,
          responseHeader: false,
          compact: true,
        ),
      );

      _ingestionDio.interceptors.add(
        PrettyDioLogger(
          requestHeader: true,
          requestBody: true,
          responseBody: true,
          responseHeader: false,
          compact: true,
        ),
      );
    }
  }

  Dio get gateway => _gatewayDio;
  Dio get recommendation => _recommendationDio;
  Dio get ingestion => _ingestionDio;
}

final dioServiceProvider = Provider<DioService>((ref) => DioService());
