import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:geolocator/geolocator.dart';
import 'package:navswap_app/features/customer/domain/entities/recommendation_entity.dart';
import '../providers/customer_providers.dart';

class CustomerHomeScreen extends ConsumerStatefulWidget {
  const CustomerHomeScreen({Key? key}) : super(key: key);

  @override
  ConsumerState<CustomerHomeScreen> createState() => _CustomerHomeScreenState();
}

class _CustomerHomeScreenState extends ConsumerState<CustomerHomeScreen> {
  late Position _currentPosition;
  bool _locationFetched = false;

  @override
  void initState() {
    super.initState();
    _getCurrentLocation();
  }

  Future<void> _getCurrentLocation() async {
    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      setState(() {
        _currentPosition = position;
        _locationFetched = true;
      });

      // Fetch recommendations after getting location
      _fetchRecommendations();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error getting location: $e')),
      );
    }
  }

  void _fetchRecommendations() {
    ref.read(recommendationProvider.notifier).fetchRecommendations(
          userId: 'user_123', // Replace with actual user ID
          latitude: _currentPosition.latitude,
          longitude: _currentPosition.longitude,
          vehicleType: 'Tesla',
          batteryLevel: 20,
        );
  }

  @override
  Widget build(BuildContext context) {
    final recommendationState = ref.watch(recommendationProvider);

    if (!_locationFetched) {
      return Scaffold(
        appBar: AppBar(title: const Text('Finding Charging Stations')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Recommended Stations'),
        elevation: 0,
      ),
      body: recommendationState.isLoading
          ? const Center(child: CircularProgressIndicator())
          : recommendationState.error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Error: ${recommendationState.error}'),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _fetchRecommendations,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : recommendationState.data?.stations.isEmpty ?? true
                  ? const Center(child: Text('No stations found'))
                  : ListView.builder(
                      itemCount: recommendationState.data?.stations.length ?? 0,
                      padding: const EdgeInsets.all(12),
                      itemBuilder: (context, index) {
                        final station =
                            recommendationState.data!.stations[index];
                        return StationCard(
                          station: station,
                          onSelect: () {
                            ref
                                .read(recommendationProvider.notifier)
                                .selectStation(station.stationId);
                            Navigator.of(context).pushNamed(
                              'stationDetail',
                              arguments: station.stationId,
                            );
                          },
                        );
                      },
                    ),
      floatingActionButton: FloatingActionButton(
        onPressed: _fetchRecommendations,
        child: const Icon(Icons.refresh),
      ),
    );
  }
}

class StationCard extends StatelessWidget {
  final StationEntity station;
  final VoidCallback onSelect;

  const StationCard({
    Key? key,
    required this.station,
    required this.onSelect,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            color: station.score > 0.7 ? Colors.green : Colors.orange,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(
            child: Text(
              '${(station.score * 100).toStringAsFixed(0)}%',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        title: Text(station.stationName),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '⏱ ${station.estimatedWaitTime} mins wait',
              style: const TextStyle(fontSize: 12),
            ),
            Text(
              '📍 ${station.estimatedDistance.toStringAsFixed(1)} km away',
              style: const TextStyle(fontSize: 12),
            ),
          ],
        ),
        trailing: Icon(
          Icons.arrow_forward_ios,
          size: 16,
          color: Colors.grey[400],
        ),
        onTap: onSelect,
      ),
    );
  }
}
