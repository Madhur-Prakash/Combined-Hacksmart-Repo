import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';

class InventoryScreen extends ConsumerWidget {
  const InventoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Battery Inventory'),
        actions: [
          IconButton(
            icon: const Icon(Icons.filter_list),
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
              // Inventory Summary
              Container(
                padding: const EdgeInsets.all(24),
                decoration: AppTheme.gradientCard,
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildInventoryStat(context, '45', 'Charged', Icons.battery_charging_full),
                        Container(
                          width: 1,
                          height: 50,
                          color: Colors.white.withOpacity(0.3),
                        ),
                        _buildInventoryStat(context, '8', 'Charging', Icons.battery_std),
                        Container(
                          width: 1,
                          height: 50,
                          color: Colors.white.withOpacity(0.3),
                        ),
                        _buildInventoryStat(context, '3', 'Faulty', Icons.battery_alert),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Divider(color: Colors.white.withOpacity(0.2)),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Total Capacity:',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: Colors.white.withOpacity(0.8),
                          ),
                        ),
                        Text(
                          '60 batteries',
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Low Stock Alert
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppTheme.warningYellow.withOpacity(0.1),
                  border: Border.all(color: AppTheme.warningYellow),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_amber_rounded, color: AppTheme.warningYellow),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Low Stock Alert: Request transporter delivery',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Charged Batteries',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 16,
                  childAspectRatio: 1.1,
                ),
                itemCount: 6,
                itemBuilder: (context, index) {
                  return _buildBatteryCard(
                    context,
                    'BAT-${1001 + index}',
                    100,
                    AppTheme.successGreen,
                    'Ready',
                    'Bay ${index + 1}',
                  );
                },
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Charging',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              ...List.generate(3, (index) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildChargingCard(
                    context,
                    'BAT-${2001 + index}',
                    65 + (index * 10),
                    '${25 - (index * 5)} min',
                  ),
                );
              }),
              
              const SizedBox(height: 24),
              
              Text(
                'Faulty Batteries',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              ...List.generate(2, (index) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildFaultyCard(
                    context,
                    'BAT-${3001 + index}',
                    index == 0 ? 'Cell Degradation' : 'Thermal Issue',
                  ),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInventoryStat(BuildContext context, String value, String label, IconData icon) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 32),
        const SizedBox(height: 8),
        Text(
          value,
          style: Theme.of(context).textTheme.displaySmall?.copyWith(
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

  Widget _buildBatteryCard(BuildContext context, String id, int charge, Color color, String status, String location) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.battery_charging_full, color: color, size: 48),
          const SizedBox(height: 12),
          Text(
            id,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          Text(
            '$charge%',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              status,
              style: TextStyle(fontSize: 12, color: color),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            location,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Widget _buildChargingCard(BuildContext context, String id, int progress, String timeLeft) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.infoBlue.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.battery_std, color: AppTheme.infoBlue, size: 32),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  id,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: progress / 100,
                    minHeight: 8,
                    backgroundColor: AppTheme.surfaceColor,
                    valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.infoBlue),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      '$progress%',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.infoBlue,
                      ),
                    ),
                    Text(
                      timeLeft,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFaultyCard(BuildContext context, String id, String issue) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: AppTheme.darkCard,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.criticalRed.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.battery_alert, color: AppTheme.criticalRed, size: 32),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  id,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  issue,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: AppTheme.criticalRed,
                  ),
                ),
              ],
            ),
          ),
          TextButton(
            onPressed: () {},
            child: const Text('Report'),
          ),
        ],
      ),
    );
  }
}
