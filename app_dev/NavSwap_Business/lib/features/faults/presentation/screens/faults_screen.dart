import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';
import '../providers/fault_providers.dart';

class FaultsScreen extends ConsumerWidget {
  const FaultsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Listen to report fault state changes
    ref.listen<AsyncValue<dynamic>>(reportFaultProvider, (previous, next) {
      next.when(
        data: (data) {
          if (data != null && data.success) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Row(
                  children: [
                    const Icon(
                      Icons.check_circle,
                      color: Colors.white,
                      size: 24,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Success!',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                          Text(
                            data.message ?? 'Fault reported successfully',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                backgroundColor: AppTheme.successGreen,
                duration: const Duration(seconds: 4),
                behavior: SnackBarBehavior.floating,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                action: SnackBarAction(
                  label: 'OK',
                  textColor: Colors.white,
                  onPressed: () {},
                ),
              ),
            );
            ref.read(reportFaultProvider.notifier).reset();
          }
        },
        loading: () {},
        error: (error, stack) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: Colors.white,
                    size: 24,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Error',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        Text(
                          'Failed to report fault: $error',
                          style: const TextStyle(fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              backgroundColor: AppTheme.criticalRed,
              duration: const Duration(seconds: 4),
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          );
        },
      );
    });

    return Scaffold(
      appBar: AppBar(
        title: const Text('Fault Monitoring'),
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
              // Fault Summary
              Container(
                padding: const EdgeInsets.all(24),
                decoration: AppTheme.darkCard,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildFaultStat(
                        context, '2', 'Critical', AppTheme.criticalRed),
                    Container(
                      width: 1,
                      height: 50,
                      color: AppTheme.surfaceColor,
                    ),
                    _buildFaultStat(
                        context, '5', 'Warning', AppTheme.warningYellow),
                    Container(
                      width: 1,
                      height: 50,
                      color: AppTheme.surfaceColor,
                    ),
                    _buildFaultStat(
                        context, '12', 'Resolved', AppTheme.successGreen),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Critical Faults',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppTheme.criticalRed.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      '2 Active',
                      style: TextStyle(
                        color: AppTheme.criticalRed,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              _buildFaultCard(
                context,
                ref,
                'Charger Bay 3 Failure',
                'Critical',
                AppTheme.criticalRed,
                'Hardware',
                'Complete charging system failure detected',
                'AI Diagnosis: Power supply module replacement required',
                'Immediately disconnect and tag the bay. Contact maintenance team.',
                {
                  'stationId': 'bay-3',
                  'faultType': 'charger_failure',
                  'severity': 'critical'
                },
              ),

              const SizedBox(height: 12),

              _buildFaultCard(
                context,
                ref,
                'Emergency Stop Triggered',
                'Critical',
                AppTheme.criticalRed,
                'Safety',
                'Emergency shutdown in Bay 5',
                'AI Diagnosis: Manual intervention detected',
                'Verify safety protocols before restart.',
                {
                  'stationId': 'bay-5',
                  'faultType': 'emergency_stop',
                  'severity': 'critical'
                },
              ),

              const SizedBox(height: 24),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Warning',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppTheme.warningYellow.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      '5 Active',
                      style: TextStyle(
                        color: AppTheme.warningYellow,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              _buildFaultCard(
                context,
                ref,
                'Temperature Elevation',
                'Warning',
                AppTheme.warningYellow,
                'Environmental',
                'Bay 7 operating at 42°C',
                'AI Diagnosis: Check ventilation system',
                'Monitor temperature. Schedule inspection.',
                {
                  'stationId': 'bay-7',
                  'faultType': 'temperature_high',
                  'severity': 'warning'
                },
              ),

              const SizedBox(height: 12),

              _buildFaultCard(
                context,
                ref,
                'Battery Cell Imbalance',
                'Warning',
                AppTheme.warningYellow,
                'Battery',
                'BAT-1045 showing voltage variance',
                'AI Diagnosis: Possible cell degradation',
                'Run diagnostic test. Consider replacement.',
                {
                  'stationId': 'station-1',
                  'batteryId': 'BAT-1045',
                  'faultType': 'cell_imbalance',
                  'severity': 'warning'
                },
              ),

              const SizedBox(height: 12),

              _buildFaultCard(
                context,
                ref,
                'Network Latency',
                'Warning',
                AppTheme.warningYellow,
                'Connectivity',
                'Intermittent connection delays',
                'AI Diagnosis: Router performance degradation',
                'Check network equipment.',
                {
                  'stationId': 'station-1',
                  'faultType': 'network_latency',
                  'severity': 'warning'
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFaultStat(
      BuildContext context, String value, String label, Color color) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: Text(
            value,
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }

  Widget _buildFaultCard(
    BuildContext context,
    WidgetRef ref,
    String title,
    String severity,
    Color color,
    String type,
    String description,
    String aiDiagnosis,
    String action,
    Map<String, dynamic> faultData,
  ) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.cardBackground,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 2),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.2),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  severity.toUpperCase(),
                  style: TextStyle(
                    color: color,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppTheme.surfaceColor,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              type,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            description,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.infoBlue.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                const Icon(Icons.auto_awesome,
                    color: AppTheme.infoBlue, size: 20),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    aiDiagnosis,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppTheme.textPrimary,
                        ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.surfaceColor,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                const Icon(Icons.lightbulb_outline,
                    color: AppTheme.warningYellow, size: 20),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    action,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    // Report fault to backend
                    ref
                        .read(reportFaultProvider.notifier)
                        .reportFault(faultData);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: color,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  child: const Text('Report Fault'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton(
                  onPressed: () {},
                  style: OutlinedButton.styleFrom(
                    side: BorderSide(color: color),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  child: Text('Details', style: TextStyle(color: color)),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
