import 'dart:convert';
import 'package:http/http.dart' as http;
import '../domain/models/verify_response_model.dart';
import '../domain/models/swap_response_model.dart';

class ScannerApiService {
  static const String baseUrl =
      'http://ec2-52-89-235-59.us-west-2.compute.amazonaws.com:3002';

  Future<VerifyResponseModel?> verifyQRCode(String qrData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/queue/verify'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'qrData': qrData}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return VerifyResponseModel.fromJson(data['data'] ?? data);
      }
      return null;
    } catch (e) {
      print('Error verifying QR code: $e');
      rethrow;
    }
  }

  Future<SwapResponseModel?> completeSwap({
    required String userId,
    required String currentBatteryId,
    required String assignedBatteryId,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/queue/swap'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'userId': userId,
          'currentBatteryId': currentBatteryId,
          'assignedBatteryId': assignedBatteryId,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return SwapResponseModel.fromJson(data['data'] ?? data);
      }
      return null;
    } catch (e) {
      print('Error completing swap: $e');
      rethrow;
    }
  }
}
