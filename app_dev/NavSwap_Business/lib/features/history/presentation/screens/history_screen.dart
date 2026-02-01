import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import '../../../../core/theme/app_theme.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Work History'),
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: AppTheme.gradientStart,
          labelColor: AppTheme.gradientStart,
          unselectedLabelColor: AppTheme.textSecondary,
          tabs: const [
            Tab(text: 'Swaps'),
            Tab(text: 'Batteries'),
            Tab(text: 'Faults'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildSwapsHistory(),
          _buildBatteriesHistory(),
          _buildFaultsHistory(),
        ],
      ),
    );
  }

  Widget _buildSwapsHistory() {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary Card
            Container(
              padding: const EdgeInsets.all(24),
              decoration: AppTheme.gradientCard,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildSummaryStat(context, '47', 'Today', Icons.today),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '324', 'This Week', Icons.calendar_view_week),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '1,245', 'Total', Icons.analytics),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            Text(
              'Recent Swaps',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            
            const SizedBox(height: 16),
            
            ...List.generate(10, (index) {
              final time = DateTime.now().subtract(Duration(hours: index));
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: _buildSwapHistoryCard(
                  context,
                  'SWAP-${2634 - index}',
                  time,
                  '${10 + index} min',
                  'BAT-${1045 - index}',
                  'Bay ${(index % 8) + 1}',
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  Widget _buildBatteriesHistory() {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary Card
            Container(
              padding: const EdgeInsets.all(24),
              decoration: AppTheme.gradientCard,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildSummaryStat(context, '52', 'Handled', Icons.battery_charging_full),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '3', 'Flagged', Icons.flag),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '8', 'Scanned', Icons.qr_code_scanner),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            Text(
              'Battery Operations',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            
            const SizedBox(height: 16),
            
            ...List.generate(8, (index) {
              final time = DateTime.now().subtract(Duration(hours: index * 2));
              final operations = ['Scanned', 'Installed', 'Removed', 'Charged'];
              final colors = [AppTheme.infoBlue, AppTheme.successGreen, AppTheme.warningYellow, AppTheme.gradientStart];
              final opIndex = index % operations.length;
              
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: _buildBatteryHistoryCard(
                  context,
                  'BAT-${1050 - index}',
                  operations[opIndex],
                  time,
                  colors[opIndex],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  Widget _buildFaultsHistory() {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary Card
            Container(
              padding: const EdgeInsets.all(24),
              decoration: AppTheme.gradientCard,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildSummaryStat(context, '3', 'Resolved', Icons.check_circle),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '18 min', 'Avg Time', Icons.schedule),
                  Container(
                    width: 1,
                    height: 40,
                    color: Colors.white.withOpacity(0.3),
                  ),
                  _buildSummaryStat(context, '100%', 'Success', Icons.star),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            Text(
              'Fault Resolution History',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            
            const SizedBox(height: 16),
            
            ...List.generate(5, (index) {
              final time = DateTime.now().subtract(Duration(hours: index * 3));
              final faults = [
                'Temperature Warning',
                'Network Latency',
                'Battery Cell Imbalance',
                'Charger Error',
                'System Timeout',
              ];
              
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: _buildFaultHistoryCard(
                  context,
                  faults[index % faults.length],
                  'Resolved',
                  time,
                  '${15 + index * 5} min',
                  AppTheme.successGreen,
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryStat(BuildContext context, String value, String label, IconData icon) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 24),
        const SizedBox(height: 8),
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
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

  Widget _buildSwapHistoryCard(
    BuildContext context,
    String swapId,
    DateTime time,
    String duration,
    String batteryId,
    String bay,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.successGreen.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.swap_horizontal_circle,
              color: AppTheme.successGreen,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  swapId,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  DateFormat('MMM dd, hh:mm a').format(time),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Text(
                      batteryId,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.infoBlue,
                      ),
                    ),
                    Text(
                      ' • $bay',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.successGreen.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'COMPLETED',
                  style: TextStyle(
                    fontSize: 10,
                    color: AppTheme.successGreen,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 4),
              Text(
                duration,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBatteryHistoryCard(
    BuildContext context,
    String batteryId,
    String operation,
    DateTime time,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.battery_charging_full,
              color: color,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  batteryId,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  DateFormat('MMM dd, hh:mm a').format(time),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              operation,
              style: TextStyle(
                fontSize: 12,
                color: color,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFaultHistoryCard(
    BuildContext context,
    String fault,
    String status,
    DateTime time,
    String resolutionTime,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.check_circle,
              color: color,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  fault,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  DateFormat('MMM dd, hh:mm a').format(time),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 4),
                Text(
                  'Resolution time: $resolutionTime',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: color,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              status.toUpperCase(),
              style: TextStyle(
                fontSize: 10,
                color: color,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
