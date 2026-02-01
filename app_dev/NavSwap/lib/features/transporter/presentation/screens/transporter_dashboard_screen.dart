import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class TransporterDashboardScreen extends ConsumerStatefulWidget {
  const TransporterDashboardScreen({super.key});

  @override
  ConsumerState<TransporterDashboardScreen> createState() => _TransporterDashboardScreenState();
}

class _TransporterDashboardScreenState extends ConsumerState<TransporterDashboardScreen> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            SliverAppBar(
              floating: true,
              backgroundColor: Colors.transparent,
              elevation: 0,
              title: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Welcome Back', style: Theme.of(context).textTheme.bodySmall),
                  Text('Transporter', style: Theme.of(context).textTheme.headlineSmall),
                ],
              ),
              actions: [
                IconButton(icon: const Icon(Icons.notifications_outlined), onPressed: () {}),
              ],
            ),
            
            SliverPadding(
              padding: const EdgeInsets.all(16),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  _buildStatsCard(),
                  const SizedBox(height: 24),
                  Text('Active Tasks', style: Theme.of(context).textTheme.headlineSmall),
                  const SizedBox(height: 12),
                  _buildTaskCard(),
                ]),
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  Widget _buildStatsCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF4A90E2), Color(0xFF6BA5E7)]),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _statItem('Credits Today', '₹2,450'),
              Container(width: 1, height: 40, color: Colors.white30),
              _statItem('Deliveries', '12'),
              Container(width: 1, height: 40, color: Colors.white30),
              _statItem('Efficiency', '94%'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _statItem(String label, String value) {
    return Column(
      children: [
        Text(value, style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }

  Widget _buildTaskCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Task #1234', style: TextStyle(fontWeight: FontWeight.bold)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFFFF6B4A).withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Text('In Progress', style: TextStyle(color: Color(0xFFFF6B4A), fontSize: 12)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Row(children: [Icon(Icons.arrow_upward, size: 16), SizedBox(width: 4), Text('Pickup: Station Alpha')]),
          const SizedBox(height: 8),
          const Row(children: [Icon(Icons.arrow_downward, size: 16), SizedBox(width: 4), Text('Drop: Station Beta')]),
          const SizedBox(height: 12),
          LinearProgressIndicator(value: 0.6, backgroundColor: Colors.grey[200], valueColor: const AlwaysStoppedAnimation(Color(0xFF4CAF50))),
          const SizedBox(height: 8),
          const Text('60% Complete', style: TextStyle(fontSize: 12, color: Color(0xFF6B6B6B))),
        ],
      ),
    );
  }

  Widget _buildBottomNav() {
    return Container(
      decoration: BoxDecoration(color: Colors.white, boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20)]),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _navItem(Icons.dashboard, 'Dashboard', 0),
              _navItem(Icons.task, 'Tasks', 1),
              _navItem(Icons.history, 'History', 2),
              _navItem(Icons.person, 'Profile', 3),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navItem(IconData icon, String label, int index) {
    final isSelected = _selectedIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _selectedIndex = index),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF1A1A1A) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Icon(icon, color: isSelected ? Colors.white : const Color(0xFF6B6B6B), size: 24),
            if (isSelected) ...[const SizedBox(width: 8), Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600))],
          ],
        ),
      ),
    );
  }
}
