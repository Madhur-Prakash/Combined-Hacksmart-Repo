import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/ticket_api_service.dart';
import '../../domain/models/ticket_response_model.dart';

final ticketApiServiceProvider = Provider((ref) => TicketApiService());

final raiseTicketProvider = StateNotifierProvider<RaiseTicketNotifier,
    AsyncValue<TicketResponseModel?>>((ref) {
  return RaiseTicketNotifier(ref.watch(ticketApiServiceProvider));
});

class RaiseTicketNotifier
    extends StateNotifier<AsyncValue<TicketResponseModel?>> {
  final TicketApiService _apiService;

  RaiseTicketNotifier(this._apiService) : super(const AsyncValue.data(null));

  Future<void> raiseTicket(Map<String, dynamic> ticketData) async {
    state = const AsyncValue.loading();
    try {
      final result = await _apiService.raiseTicket(ticketData);
      state = AsyncValue.data(result);
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }

  void reset() {
    state = const AsyncValue.data(null);
  }
}
