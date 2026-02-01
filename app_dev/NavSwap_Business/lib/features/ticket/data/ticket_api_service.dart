import 'dart:convert';
import 'package:http/http.dart' as http;
import '../domain/models/ticket_response_model.dart';

class TicketApiService {
  static const String baseUrl =
      'http://ec2-52-89-235-59.us-west-2.compute.amazonaws.com:3002';

  Future<TicketResponseModel?> raiseTicket(
      Map<String, dynamic> ticketData) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/ticket/manual'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(ticketData),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return TicketResponseModel.fromJson(data['data'] ?? data);
      }
      return null;
    } catch (e) {
      print('Error raising ticket: $e');
      rethrow;
    }
  }
}
