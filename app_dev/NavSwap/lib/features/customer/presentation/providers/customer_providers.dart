import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/services/dio_service.dart';
import '../../data/datasources/customer_remote_datasource.dart';
import '../../data/repositories/customer_repository.dart';
import '../../domain/usecases/get_recommendations_usecase.dart';
import '../../domain/entities/recommendation_entity.dart';

// DataSource Provider
final customerRemoteDataSourceProvider =
    Provider<CustomerRemoteDataSource>((ref) {
  final dioService = ref.watch(dioServiceProvider);
  return CustomerRemoteDataSourceImpl(dioService);
});

// Repository Provider
final customerRepositoryProvider = Provider<CustomerRepository>((ref) {
  final dataSource = ref.watch(customerRemoteDataSourceProvider);
  return CustomerRepositoryImpl(dataSource);
});

// UseCases Providers
final getRecommendationsUseCaseProvider =
    Provider<GetRecommendationsUseCase>((ref) {
  final repository = ref.watch(customerRepositoryProvider);
  return GetRecommendationsUseCase(repository);
});

final recordSelectionUseCaseProvider = Provider<RecordSelectionUseCase>((ref) {
  final repository = ref.watch(customerRepositoryProvider);
  return RecordSelectionUseCase(repository);
});

final submitFeedbackUseCaseProvider = Provider<SubmitFeedbackUseCase>((ref) {
  final repository = ref.watch(customerRepositoryProvider);
  return SubmitFeedbackUseCase(repository);
});

final getStationScoreUseCaseProvider = Provider<GetStationScoreUseCase>((ref) {
  final repository = ref.watch(customerRepositoryProvider);
  return GetStationScoreUseCase(repository);
});

final getStationHealthUseCaseProvider =
    Provider<GetStationHealthUseCase>((ref) {
  final repository = ref.watch(customerRepositoryProvider);
  return GetStationHealthUseCase(repository);
});

// State for recommendations
class RecommendationState {
  final RecommendationEntity? data;
  final bool isLoading;
  final String? error;
  final String? lastRequestId;

  RecommendationState({
    this.data,
    this.isLoading = false,
    this.error,
    this.lastRequestId,
  });

  RecommendationState copyWith({
    RecommendationEntity? data,
    bool? isLoading,
    String? error,
    String? lastRequestId,
  }) {
    return RecommendationState(
      data: data ?? this.data,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
      lastRequestId: lastRequestId ?? this.lastRequestId,
    );
  }
}

// Recommendation Notifier
class RecommendationNotifier extends StateNotifier<RecommendationState> {
  final GetRecommendationsUseCase getRecommendationsUseCase;
  final RecordSelectionUseCase recordSelectionUseCase;
  final SubmitFeedbackUseCase submitFeedbackUseCase;

  RecommendationNotifier({
    required this.getRecommendationsUseCase,
    required this.recordSelectionUseCase,
    required this.submitFeedbackUseCase,
  }) : super(RecommendationState());

  Future<void> fetchRecommendations({
    required String userId,
    required double latitude,
    required double longitude,
    String? vehicleType,
    int? batteryLevel,
  }) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final result = await getRecommendationsUseCase(
        userId: userId,
        latitude: latitude,
        longitude: longitude,
        vehicleType: vehicleType,
        batteryLevel: batteryLevel,
      );

      state = state.copyWith(
        data: result,
        isLoading: false,
        lastRequestId: result.requestId,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  Future<void> selectStation(String stationId) async {
    final requestId = state.lastRequestId;
    if (requestId == null) return;

    try {
      await recordSelectionUseCase(
        requestId: requestId,
        stationId: stationId,
      );
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }

  Future<void> submitFeedback(int rating) async {
    final requestId = state.lastRequestId;
    if (requestId == null) return;

    try {
      await submitFeedbackUseCase(
        requestId: requestId,
        rating: rating,
      );
    } catch (e) {
      state = state.copyWith(error: e.toString());
    }
  }
}

// Recommendation Provider
final recommendationProvider =
    StateNotifierProvider<RecommendationNotifier, RecommendationState>((ref) {
  final getRecommendations = ref.watch(getRecommendationsUseCaseProvider);
  final recordSelection = ref.watch(recordSelectionUseCaseProvider);
  final submitFeedback = ref.watch(submitFeedbackUseCaseProvider);

  return RecommendationNotifier(
    getRecommendationsUseCase: getRecommendations,
    recordSelectionUseCase: recordSelection,
    submitFeedbackUseCase: submitFeedback,
  );
});

// Station Score Provider
final stationScoreProvider =
    FutureProvider.family((ref, String stationId) async {
  final useCase = ref.watch(getStationScoreUseCaseProvider);
  return useCase(stationId);
});

// Station Health Provider
final stationHealthProvider =
    FutureProvider.family((ref, String stationId) async {
  final useCase = ref.watch(getStationHealthUseCaseProvider);
  return useCase(stationId);
});
