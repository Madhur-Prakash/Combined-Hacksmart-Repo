import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/scanner_api_service.dart';
import '../../domain/models/verify_response_model.dart';
import '../../domain/models/swap_response_model.dart';

final scannerApiServiceProvider = Provider((ref) => ScannerApiService());

final verifyQRProvider =
    StateNotifierProvider<VerifyQRNotifier, AsyncValue<VerifyResponseModel?>>(
        (ref) {
  return VerifyQRNotifier(ref.watch(scannerApiServiceProvider));
});

class VerifyQRNotifier extends StateNotifier<AsyncValue<VerifyResponseModel?>> {
  final ScannerApiService _apiService;

  VerifyQRNotifier(this._apiService) : super(const AsyncValue.data(null));

  Future<void> verifyQRCode(String qrData) async {
    state = const AsyncValue.loading();
    try {
      final result = await _apiService.verifyQRCode(qrData);
      state = AsyncValue.data(result);
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }

  void reset() {
    state = const AsyncValue.data(null);
  }
}

final completeSwapProvider =
    StateNotifierProvider<CompleteSwapNotifier, AsyncValue<SwapResponseModel?>>(
        (ref) {
  return CompleteSwapNotifier(ref.watch(scannerApiServiceProvider));
});

class CompleteSwapNotifier
    extends StateNotifier<AsyncValue<SwapResponseModel?>> {
  final ScannerApiService _apiService;

  CompleteSwapNotifier(this._apiService) : super(const AsyncValue.data(null));

  Future<void> completeSwap({
    required String userId,
    required String currentBatteryId,
    required String assignedBatteryId,
  }) async {
    state = const AsyncValue.loading();
    try {
      final result = await _apiService.completeSwap(
        userId: userId,
        currentBatteryId: currentBatteryId,
        assignedBatteryId: assignedBatteryId,
      );
      state = AsyncValue.data(result);
    } catch (e, stack) {
      state = AsyncValue.error(e, stack);
    }
  }

  void reset() {
    state = const AsyncValue.data(null);
  }
}
