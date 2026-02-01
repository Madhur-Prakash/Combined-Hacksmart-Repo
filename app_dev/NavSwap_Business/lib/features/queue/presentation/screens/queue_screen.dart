import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';

class QueueScreen extends ConsumerWidget {
  const QueueScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Queue Monitor'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {},
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Queue Summary
              Container(
                padding: const EdgeInsets.all(24),
                decoration: AppTheme.gradientCard,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildQueueStat(context, '8', 'In Queue', Icons.people),
                    Container(
                      width: 1,
                      height: 40,
                      color: Colors.white.withOpacity(0.3),
                    ),
                    _buildQueueStat(context, '3', 'In Progress', Icons.autorenew),
                    Container(
                      width: 1,
                      height: 40,
                      color: Colors.white.withOpacity(0.3),
                    ),
                    _buildQueueStat(context, '12 min', 'Avg Wait', Icons.schedule),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Active Swaps',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              _buildSwapCard(
                context,
                'BAY-001',
                'Swap in Progress',
                'ETA: 3 min',
                AppTheme.successGreen,
                85,
              ),
              
              const SizedBox(height: 12),
              
              _buildSwapCard(
                context,
                'BAY-003',
                'Swap in Progress',
                'ETA: 5 min',
                AppTheme.successGreen,
                60,
              ),
              
              const SizedBox(height: 12),
              
              _buildSwapCard(
                context,
                'BAY-005',
                'Battery Removal',
                'ETA: 8 min',
                AppTheme.infoBlue,
                30,
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Waiting Queue',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              ...List.generate(5, (index) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildQueueItem(
                    context,
                    'Customer ${index + 1}',
                    'Vehicle: MH-02-XX-${1234 + index}',
                    'Position: ${index + 1}',
                    '${5 + index * 2} min',
                  ),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQueueStat(BuildContext context, String value, String label, IconData icon) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 28),
        const SizedBox(height: 8),
        Text(
          value,
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.white.withOpacity(0.8),
          ),
        ),
      ],
    );
  }

  Widget _buildSwapCard(BuildContext context, String bay, String status, String eta, Color color, double progress) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: AppTheme.darkCard,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(Icons.ev_station, color: color, size: 24),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        bay,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      Text(
                        status,
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(color: color),
                      ),
                    ],
                  ),
                ],
              ),
              Text(
                eta,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: AppTheme.textSecondary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progress / 100,
              minHeight: 8,
              backgroundColor: AppTheme.surfaceColor,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQueueItem(BuildContext context, String customer, String vehicle, String position, String wait) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppTheme.surfaceColor,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.person_outline, color: AppTheme.textSecondary),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  customer,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  vehicle,
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                position,
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              Text(
                'Wait: $wait',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: AppTheme.warningYellow,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
