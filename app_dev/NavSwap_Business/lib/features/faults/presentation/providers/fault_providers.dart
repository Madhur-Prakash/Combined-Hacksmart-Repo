import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/fault_api_service.dart';
import '../../domain/models/fault_report_model.dart';

final faultApiServiceProvider = Provider((ref) => FaultApiService());

final reportFaultProvider =
    StateNotifierProvider<ReportFaultNotifier, AsyncValue<FaultReportModel?>>(
        (ref) {
  return ReportFaultNotifier(ref.watch(faultApiServiceProvider));
});

class ReportFaultNotifier extends StateNotifier<AsyncValue<FaultReportModel?>> {
  final FaultApiService _apiService;

  ReportFaultNotifier(this._apiService) : super(const AsyncValue.data(null));

  Future<void> reportFault(Map<String, dynamic> faultData) async {
    state = const AsyncValue.loading();
    try {
      final result = await _apiService.reportFault(faultData);
      state = AsyncValue.data(result);
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }

  void reset() {
    state = const AsyncValue.data(null);
  }
}
