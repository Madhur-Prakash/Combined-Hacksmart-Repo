import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shimmer/shimmer.dart'; // Add this line
import '../providers/customer_providers.dart';

class StationDetailScreen extends ConsumerWidget {
  final String stationId;

  const StationDetailScreen({
    Key? key,
    required this.stationId,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final scoreAsync = ref.watch(stationScoreProvider(stationId));
    final healthAsync = ref.watch(stationHealthProvider(stationId));

    return Scaffold(
      appBar: AppBar(title: Text('Station $stationId Details')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Score Section
            const Text(
              'Performance Score',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            scoreAsync.when(
              data: (score) => _buildScoreCard(score),
              loading: () => Shimmer.fromColors(
                baseColor: Colors.grey,
                highlightColor: Colors.white30,
                child: SizedBox(height: 120, width: double.infinity),
              ),
              error: (e, st) => Text('Error: $e'),
            ),
            const SizedBox(height: 24),
            // Health Section
            const Text(
              'Station Health',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            healthAsync.when(
              data: (health) => _buildHealthCard(health),
              loading: () => Shimmer.fromColors(
                baseColor: Colors.grey,
                highlightColor: Colors.white30,
                child: SizedBox(height: 100, width: double.infinity),
              ),
              error: (e, st) => Text('Error: $e'),
            ),
            const SizedBox(height: 32),
            // Action Buttons
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.of(context).pushNamed('queue');
                },
                child: const Padding(
                  padding: EdgeInsets.all(12),
                  child: Text('Join Queue'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildScoreCard(dynamic score) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _scoreItem('Overall', score['overallScore'] ?? 0),
              _scoreItem('Wait Time', score['waitTimeScore'] ?? 0),
              _scoreItem('Reliability', score['reliabilityScore'] ?? 0),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _scoreItem('Energy', score['energyStabilityScore'] ?? 0),
              _scoreItem('Chargers', score['chargerAvailabilityScore'] ?? 0),
            ],
          ),
        ],
      ),
    );
  }

  Widget _scoreItem(String label, dynamic value) {
    final doubleValue = (value as num).toDouble();
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 12)),
        const SizedBox(height: 4),
        Text(
          '${(doubleValue * 100).toStringAsFixed(0)}%',
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
        ),
      ],
    );
  }

  Widget _buildHealthCard(dynamic health) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: health['status'] == 'operational'
            ? Colors.green.shade50
            : Colors.red.shade50,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: health['status'] == 'operational'
                      ? Colors.green
                      : Colors.red,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                health['status'] ?? 'Unknown',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text('Health Score: ${health['healthScore'] ?? 0}%'),
          if (health['activeAlerts'] != null &&
              health['activeAlerts'].isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Active Alerts:'),
                  ...?health['activeAlerts']?.map<Widget>((alert) {
                    return Text('• $alert',
                        style: const TextStyle(fontSize: 12));
                  }).toList(),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
