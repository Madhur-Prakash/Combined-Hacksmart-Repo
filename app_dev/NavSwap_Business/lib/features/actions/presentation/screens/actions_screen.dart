import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';

class ActionsScreen extends ConsumerWidget {
  const ActionsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Recommendations'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // AI Header
              Container(
                padding: const EdgeInsets.all(24),
                decoration: AppTheme.gradientCard,
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Icon(
                        Icons.auto_awesome,
                        color: Colors.white,
                        size: 32,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'AI Assistant',
                            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '8 active recommendations',
                            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: Colors.white.withOpacity(0.9),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'High Priority',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              _buildRecommendationCard(
                context,
                'Preventive Maintenance Required',
                'High Priority',
                AppTheme.criticalRed,
                Icons.build_outlined,
                'Charger Bay 3 showing early signs of wear. Schedule maintenance to prevent failure.',
                'Expected downtime: 2 hours',
                ['Schedule Now', 'Snooze'],
              ),
              
              const SizedBox(height: 12),
              
              _buildRecommendationCard(
                context,
                'Peak Hour Staffing',
                'High Priority',
                AppTheme.warningYellow,
                Icons.people_outline,
                'Peak hours starting in 45 minutes. Consider adding 2 more staff members.',
                'Expected queue increase: 60%',
                ['Call Staff', 'Dismiss'],
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Medium Priority',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              _buildRecommendationCard(
                context,
                'Battery Reorder Alert',
                'Medium Priority',
                AppTheme.infoBlue,
                Icons.shopping_cart_outlined,
                'Current inventory below optimal level. Request 15 additional batteries.',
                'Delivery time: 24 hours',
                ['Order Now', 'Later'],
              ),
              
              const SizedBox(height: 12),
              
              _buildRecommendationCard(
                context,
                'Performance Optimization',
                'Medium Priority',
                AppTheme.infoBlue,
                Icons.trending_up,
                'Bay assignment algorithm can be optimized. Implementing new routing will reduce avg swap time by 8%.',
                'No downtime required',
                ['Apply', 'Review'],
              ),
              
              const SizedBox(height: 24),
              
              Text(
                'Low Priority',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              
              const SizedBox(height: 16),
              
              _buildRecommendationCard(
                context,
                'Energy Usage Report',
                'Low Priority',
                AppTheme.successGreen,
                Icons.energy_savings_leaf,
                'Weekly energy consumption analysis shows 12% efficiency improvement opportunity.',
                'Potential savings: ₹8,500/month',
                ['View Report', 'Dismiss'],
              ),
              
              const SizedBox(height: 12),
              
              _buildRecommendationCard(
                context,
                'Staff Training',
                'Low Priority',
                AppTheme.successGreen,
                Icons.school_outlined,
                'New safety protocol training available. Recommended for all staff.',
                'Duration: 30 minutes per staff',
                ['Schedule', 'Later'],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRecommendationCard(
    BuildContext context,
    String title,
    String priority,
    Color color,
    IconData icon,
    String description,
    String metadata,
    List<String> actions,
  ) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppTheme.cardBackground,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 24),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        priority,
                        style: TextStyle(
                          fontSize: 11,
                          color: color,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          
          Text(
            description,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          
          const SizedBox(height: 12),
          
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.surfaceColor,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                const Icon(Icons.info_outline, color: AppTheme.textSecondary, size: 16),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    metadata,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
          
          const SizedBox(height: 16),
          
          Row(
            children: actions.map((action) {
              final isFirst = actions.indexOf(action) == 0;
              return Expanded(
                child: Padding(
                  padding: EdgeInsets.only(right: isFirst ? 8 : 0, left: isFirst ? 0 : 8),
                  child: isFirst
                      ? ElevatedButton(
                          onPressed: () {},
                          style: ElevatedButton.styleFrom(
                            backgroundColor: color,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: Text(action),
                        )
                      : OutlinedButton(
                          onPressed: () {},
                          style: OutlinedButton.styleFrom(
                            side: BorderSide(color: AppTheme.textSecondary),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: Text(action),
                        ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}
