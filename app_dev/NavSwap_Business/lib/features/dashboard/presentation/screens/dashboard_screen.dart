import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';
import '../providers/station_providers.dart';

class DashboardScreen extends ConsumerStatefulWidget {
  const DashboardScreen({super.key});

  @override
  ConsumerState<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends ConsumerState<DashboardScreen> {
  int _selectedIndex = 0;
  final String stationId = 'ST_101'; // Replace with actual station ID

  @override
  Widget build(BuildContext context) {
    final scoreAsync = ref.watch(stationScoreProvider(stationId));
    final healthAsync = ref.watch(stationHealthProvider(stationId));
    final predictionsAsync = ref.watch(stationPredictionsProvider(stationId));

    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with Profile
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Station Control',
                        style: Theme.of(context).textTheme.displaySmall,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Welcome back, Rajesh Kumar',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                  GestureDetector(
                    onTap: () => context.push('/staff/profile'),
                    child: const CircleAvatar(
                      radius: 24,
                      backgroundColor: AppTheme.gradientEnd,
                      child: Icon(Icons.person, color: Colors.white),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 24),

              // Station Summary Card
              healthAsync.when(
                data: (health) => Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: AppTheme.gradientCard,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Delhi Sector 18',
                                style: Theme.of(context)
                                    .textTheme
                                    .headlineMedium
                                    ?.copyWith(
                                      color: Colors.white,
                                    ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'Station ID: ${health?.stationId ?? 'NS-DL-018'}',
                                style: Theme.of(context)
                                    .textTheme
                                    .bodyMedium
                                    ?.copyWith(
                                      color: Colors.white.withOpacity(0.8),
                                    ),
                              ),
                            ],
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  width: 8,
                                  height: 8,
                                  decoration: BoxDecoration(
                                    color: _getStatusColor(
                                        health?.status ?? 'operational'),
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  (health?.status ?? 'ACTIVE').toUpperCase(),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 24),

                      // Stats Row - Using predictions for queue data
                      predictionsAsync.when(
                        data: (predictions) => Row(
                          children: [
                            Expanded(
                              child: _buildStatItem(
                                context,
                                _calculateActiveSwaps(
                                    predictions?.loadForecast.predictedLoad ??
                                        0.0),
                                'Active Swaps',
                                Icons.swap_horizontal_circle,
                              ),
                            ),
                            Container(
                              width: 1,
                              height: 40,
                              color: Colors.white.withOpacity(0.2),
                            ),
                            Expanded(
                              child: _buildStatItem(
                                context,
                                _calculateQueue(
                                    predictions?.loadForecast.predictedLoad ??
                                        0.0),
                                'In Queue',
                                Icons.people_outline,
                              ),
                            ),
                            Container(
                              width: 1,
                              height: 40,
                              color: Colors.white.withOpacity(0.2),
                            ),
                            Expanded(
                              child: _buildStatItem(
                                context,
                                _calculateAvailable(
                                    predictions?.loadForecast.predictedLoad ??
                                        0.0),
                                'Available',
                                Icons.battery_charging_full,
                              ),
                            ),
                          ],
                        ),
                        loading: () =>
                            const Center(child: CircularProgressIndicator()),
                        error: (_, __) => Row(
                          children: [
                            Expanded(
                              child: _buildStatItem(
                                context,
                                '12',
                                'Active Swaps',
                                Icons.swap_horizontal_circle,
                              ),
                            ),
                            Container(
                              width: 1,
                              height: 40,
                              color: Colors.white.withOpacity(0.2),
                            ),
                            Expanded(
                              child: _buildStatItem(
                                context,
                                '8',
                                'In Queue',
                                Icons.people_outline,
                              ),
                            ),
                            Container(
                              width: 1,
                              height: 40,
                              color: Colors.white.withOpacity(0.2),
                            ),
                            Expanded(
                              child: _buildStatItem(
                                context,
                                '45',
                                'Available',
                                Icons.battery_charging_full,
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 16),

                      // Charger Health
                      Row(
                        children: [
                          const Icon(
                            Icons.offline_bolt,
                            color: Colors.white,
                            size: 20,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'Charger Health:',
                            style: Theme.of(context)
                                .textTheme
                                .bodyMedium
                                ?.copyWith(
                                  color: Colors.white.withOpacity(0.8),
                                ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${health?.healthScore ?? 98}%',
                            style: Theme.of(context)
                                .textTheme
                                .titleLarge
                                ?.copyWith(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (_, __) => Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: AppTheme.gradientCard,
                  child: const Text(
                    'Error loading station data',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // Quick Actions Grid
              Text(
                'Quick Actions',
                style: Theme.of(context).textTheme.headlineSmall,
              ),

              const SizedBox(height: 16),

              GridView.count(
                crossAxisCount: 2,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: 1.1,
                children: [
                  _buildActionCard(
                    context,
                    'Queue Monitor',
                    Icons.format_list_numbered,
                    AppTheme.infoBlue,
                    () => context.push('/staff/queue'),
                  ),
                  _buildActionCard(
                    context,
                    'Battery Inventory',
                    Icons.inventory_2_outlined,
                    AppTheme.successGreen,
                    () => context.push('/staff/inventory'),
                  ),
                  _buildActionCard(
                    context,
                    'Fault Alerts',
                    Icons.warning_amber_rounded,
                    AppTheme.warningYellow,
                    () => context.push('/staff/faults'),
                  ),
                  _buildActionCard(
                    context,
                    'Ticket',
                    Icons.airplane_ticket_outlined,
                    AppTheme.gradientStart,
                    () => context.push('/staff/ticket'),
                  ),
                ],
              ),

              const SizedBox(height: 24),

              // AI Recommendations
              predictionsAsync.when(
                data: (predictions) => Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.darkCard,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: AppTheme.infoBlue.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: const Icon(
                              Icons.auto_awesome,
                              color: AppTheme.infoBlue,
                              size: 20,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(
                            'AI Recommendations',
                            style: Theme.of(context).textTheme.titleLarge,
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      if (predictions?.faultPrediction.riskLevel == 'high')
                        _buildRecommendationItem(
                          context,
                          'Schedule preventive maintenance - High fault risk detected',
                          'High Priority',
                          AppTheme.warningYellow,
                        )
                      else if (predictions?.faultPrediction.riskLevel ==
                          'medium')
                        _buildRecommendationItem(
                          context,
                          'Monitor charger bay closely - Medium fault risk',
                          'Medium Priority',
                          AppTheme.infoBlue,
                        )
                      else
                        _buildRecommendationItem(
                          context,
                          'System operating normally - Low fault risk',
                          'Low Priority',
                          AppTheme.successGreen,
                        ),
                      const Divider(height: 24, color: AppTheme.surfaceColor),
                      if (predictions != null &&
                          predictions.loadForecast.predictedLoad > 0.7)
                        _buildRecommendationItem(
                          context,
                          'Peak hours approaching (${predictions.loadForecast.peakTimeStart} - ${predictions.loadForecast.peakTimeEnd}) - Consider adding staff',
                          'Medium Priority',
                          AppTheme.infoBlue,
                        ),
                      const SizedBox(height: 12),
                      TextButton(
                        onPressed: () => context.push('/staff/actions'),
                        child: const Text('View All Recommendations'),
                      ),
                    ],
                  ),
                ),
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (_, __) => Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.darkCard,
                  child: const Text('Error loading predictions'),
                ),
              ),

              const SizedBox(height: 24),

              // Today's Performance
              scoreAsync.when(
                data: (score) => Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.darkCard,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Today\'s Performance',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _buildPerformanceMetric(
                              context,
                              '${_calculateSwaps(score?.componentScores.availabilityScore ?? 0.9)}',
                              'Swaps Completed'),
                          _buildPerformanceMetric(
                              context, '3', 'Faults Resolved'),
                          _buildPerformanceMetric(
                              context,
                              '${_calculateAvgTime(score?.componentScores.waitTimeScore ?? 0.85)} min',
                              'Avg Swap Time'),
                        ],
                      ),
                    ],
                  ),
                ),
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (_, __) => Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.darkCard,
                  child: const Text('Error loading performance data'),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppTheme.cardBackground,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 16,
              offset: const Offset(0, -4),
            ),
          ],
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(context, Icons.dashboard, 'Dashboard', 0),
                _buildNavItem(context, Icons.qr_code_scanner, 'Scanner', 1),
                _buildNavItem(context, Icons.history, 'History', 2),
                _buildNavItem(context, Icons.person, 'Profile', 3),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _getStatusColor(String status) {
    switch (status.toLowerCase()) {
      case 'operational':
        return AppTheme.successGreen;
      case 'degraded':
        return AppTheme.warningYellow;
      case 'offline':
      case 'maintenance':
        return Colors.red;
      default:
        return AppTheme.successGreen;
    }
  }

  String _calculateActiveSwaps(double predictedLoad) {
    return (predictedLoad * 50).round().toString();
  }

  String _calculateQueue(double predictedLoad) {
    return (predictedLoad * 15).round().toString();
  }

  String _calculateAvailable(double predictedLoad) {
    return (100 - (predictedLoad * 100)).round().toString();
  }

  int _calculateSwaps(double availabilityScore) {
    return (availabilityScore * 52).round();
  }

  int _calculateAvgTime(double waitTimeScore) {
    return (20 - (waitTimeScore * 8)).round();
  }

  Widget _buildStatItem(
      BuildContext context, String value, String label, IconData icon) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 24),
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
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildActionCard(BuildContext context, String title, IconData icon,
      Color color, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: AppTheme.darkCard,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: color, size: 32),
            ),
            const SizedBox(height: 12),
            Text(
              title,
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationItem(
      BuildContext context, String text, String priority, Color color) {
    return Row(
      children: [
        Container(
          width: 4,
          height: 40,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                text,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 4),
              Text(
                priority,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: color),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildPerformanceMetric(
      BuildContext context, String value, String label) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                color: AppTheme.gradientStart,
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildNavItem(
      BuildContext context, IconData icon, String label, int index) {
    final isSelected = _selectedIndex == index;
    return GestureDetector(
      onTap: () {
        setState(() => _selectedIndex = index);
        switch (index) {
          case 0:
            context.go('/staff/dashboard');
            break;
          case 1:
            context.push('/staff/scanner');
            break;
          case 2:
            context.push('/staff/history');
            break;
          case 3:
            context.push('/staff/profile');
            break;
        }
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: isSelected ? AppTheme.gradientStart : AppTheme.textSecondary,
            size: 28,
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color:
                  isSelected ? AppTheme.gradientStart : AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}
