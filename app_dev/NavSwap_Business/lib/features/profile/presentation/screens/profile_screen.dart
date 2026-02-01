import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';

class ProfileScreen extends ConsumerWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {},
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // Profile Header
              Container(
                padding: const EdgeInsets.all(24),
                decoration: AppTheme.gradientCard,
                child: Column(
                  children: [
                    CircleAvatar(
                      radius: 50,
                      backgroundColor: Colors.white.withOpacity(0.2),
                      child: const Icon(
                        Icons.person,
                        color: Colors.white,
                        size: 50,
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Rajesh Kumar',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Station Operator',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white70,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: const Text(
                        'EMP-00145',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Today's Stats
              Text(
                'Today\'s Performance',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              Row(
                children: [
                  Expanded(
                    child: _buildStatCard(
                      context,
                      '47',
                      'Swaps Handled',
                      Icons.swap_horizontal_circle,
                      AppTheme.gradientStart,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildStatCard(
                      context,
                      '3',
                      'Faults Resolved',
                      Icons.check_circle,
                      AppTheme.successGreen,
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 12),
              
              Row(
                children: [
                  Expanded(
                    child: _buildStatCard(
                      context,
                      '12 min',
                      'Avg Swap Time',
                      Icons.schedule,
                      AppTheme.infoBlue,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildStatCard(
                      context,
                      '8 hrs',
                      'Shift Duration',
                      Icons.access_time,
                      AppTheme.warningYellow,
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 24),
              
              // Performance Metrics
              Container(
                padding: const EdgeInsets.all(20),
                decoration: AppTheme.darkCard,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Performance Metrics',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 20),
                    _buildMetricRow(context, 'Efficiency Score', '94%', 94, AppTheme.successGreen),
                    const SizedBox(height: 16),
                    _buildMetricRow(context, 'Response Time', '87%', 87, AppTheme.infoBlue),
                    const SizedBox(height: 16),
                    _buildMetricRow(context, 'Customer Rating', '4.8/5.0', 96, AppTheme.warningYellow),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Shift Information
              Container(
                padding: const EdgeInsets.all(20),
                decoration: AppTheme.darkCard,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Current Shift',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 16),
                    _buildInfoRow(context, Icons.access_time, 'Shift Time', '09:00 AM - 05:00 PM'),
                    const Divider(height: 24, color: AppTheme.surfaceColor),
                    _buildInfoRow(context, Icons.location_on, 'Station', 'Delhi Sector 18'),
                    const Divider(height: 24, color: AppTheme.surfaceColor),
                    _buildInfoRow(context, Icons.calendar_today, 'Date', '31 Jan 2026'),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Quick Actions
              _buildActionItem(
                context,
                Icons.history,
                'Work History',
                'View your complete work log',
                () => context.push('/staff/history'),
              ),
              
              const SizedBox(height: 12),
              
              _buildActionItem(
                context,
                Icons.help_outline,
                'Help & Support',
                'Get assistance and FAQs',
                () {},
              ),
              
              const SizedBox(height: 12),
              
              _buildActionItem(
                context,
                Icons.info_outline,
                'About',
                'App version and information',
                () {},
              ),
              
              const SizedBox(height: 24),
              
              // Logout Button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: () => context.go('/login'),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: AppTheme.criticalRed),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  child: const Text(
                    'Logout',
                    style: TextStyle(color: AppTheme.criticalRed),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard(BuildContext context, String value, String label, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: AppTheme.darkCard,
      child: Column(
        children: [
          Icon(icon, color: color, size: 32),
          const SizedBox(height: 12),
          Text(
            value,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              color: color,
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
      ),
    );
  }

  Widget _buildMetricRow(BuildContext context, String label, String value, double progress, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            Text(
              value,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(color: color),
            ),
          ],
        ),
        const SizedBox(height: 8),
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
    );
  }

  Widget _buildInfoRow(BuildContext context, IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, color: AppTheme.textSecondary, size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.titleMedium,
        ),
      ],
    );
  }

  Widget _buildActionItem(BuildContext context, IconData icon, String title, String subtitle, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: AppTheme.darkCard,
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.surfaceColor,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: AppTheme.textSecondary),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
            const Icon(Icons.arrow_forward_ios, color: AppTheme.textSecondary, size: 16),
          ],
        ),
      ),
    );
  }
}
