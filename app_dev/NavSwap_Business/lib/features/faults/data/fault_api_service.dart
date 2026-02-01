import 'dart:convert';
import 'package:http/http.dart' as http;
import '../domain/models/fault_report_model.dart';

class FaultApiService {
  static const String baseUrl =
      'http://ec2-52-89-235-59.us-west-2.compute.amazonaws.com:3002';

  Future<FaultReportModel?> reportFault(Map<String, dynamic> faultData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/fault/report'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(faultData),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return FaultReportModel.fromJson(data['data'] ?? data);
      }
      return null;
    } catch (e) {
      print('Error reporting fault: $e');
      rethrow;
    }
  }
}
