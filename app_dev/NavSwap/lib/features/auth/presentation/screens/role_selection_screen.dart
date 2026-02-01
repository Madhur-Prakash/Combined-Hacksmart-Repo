import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

class RoleSelectionScreen extends ConsumerStatefulWidget {
  const RoleSelectionScreen({super.key});

  @override
  ConsumerState<RoleSelectionScreen> createState() => _RoleSelectionScreenState();
}

class _RoleSelectionScreenState extends ConsumerState<RoleSelectionScreen> {
  String? _selectedRole;

  Future<void> _handleContinue() async {
    if (_selectedRole == null) return;

    // Navigate to Login with selected role
    context.go(Uri(path: '/auth/login', queryParameters: {'role': _selectedRole}).toString());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 40),
              
              Text(
                'Choose Your Role',
                style: Theme.of(context).textTheme.displaySmall,
              ),
              const SizedBox(height: 8),
              Text(
                'Select how you want to use NAVSWAP',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Theme.of(context).textTheme.bodySmall?.color,
                ),
              ),
              
              const SizedBox(height: 48),
              
              _RoleCard(
                title: 'Customer',
                description: 'Find stations, manage swaps, and get AI recommendations',
                icon: Icons.person,
                isSelected: _selectedRole == 'customer',
                onTap: () => setState(() => _selectedRole = 'customer'),
              ),
              
              const SizedBox(height: 16),
              
              _RoleCard(
                title: 'Transporter',
                description: 'Deliver batteries, complete tasks, and earn credits',
                icon: Icons.local_shipping,
                isSelected: _selectedRole == 'transporter',
                onTap: () => setState(() => _selectedRole = 'transporter'),
              ),
              
              const Spacer(),
              
              SizedBox(
                height: 56,
                child: ElevatedButton(
                  onPressed: _selectedRole == null ? null : _handleContinue,
                  child: const Text('Continue'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final bool isSelected;
  final VoidCallback onTap;

  const _RoleCard({
    required this.title,
    required this.description,
    required this.icon,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF1A1A1A) : Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? const Color(0xFF1A1A1A) : const Color(0xFFE8E8E8),
            width: 2,
          ),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: isSelected ? Colors.white.withOpacity(0.2) : const Color(0xFFF8F9FA),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                icon,
                size: 32,
                color: isSelected ? Colors.white : const Color(0xFF1A1A1A),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: isSelected ? Colors.white : const Color(0xFF1A1A1A),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    description,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: isSelected ? Colors.white70 : const Color(0xFF6B6B6B),
                    ),
                  ),
                ],
              ),
            ),
            if (isSelected)
              const Icon(
                Icons.check_circle,
                color: Colors.white,
              ),
          ],
        ),
      ),
    );
  }
}
