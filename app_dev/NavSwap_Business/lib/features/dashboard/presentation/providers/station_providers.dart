import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/station_api_service.dart';
import '../../domain/models/station_score_model.dart';
import '../../domain/models/station_health_model.dart';
import '../../domain/models/station_predictions_model.dart';

final stationApiServiceProvider = Provider((ref) => StationApiService());

final stationScoreProvider =
    FutureProvider.family<StationScoreModel?, String>((ref, stationId) async {
  final apiService = ref.watch(stationApiServiceProvider);
  return await apiService.getStationScore(stationId);
});

final stationHealthProvider =
    FutureProvider.family<StationHealthModel?, String>((ref, stationId) async {
  final apiService = ref.watch(stationApiServiceProvider);
  return await apiService.getStationHealth(stationId);
});

final stationPredictionsProvider =
    FutureProvider.family<StationPredictionsModel?, String>(
        (ref, stationId) async {
  final apiService = ref.watch(stationApiServiceProvider);
  return await apiService.getStationPredictions(stationId);
});
