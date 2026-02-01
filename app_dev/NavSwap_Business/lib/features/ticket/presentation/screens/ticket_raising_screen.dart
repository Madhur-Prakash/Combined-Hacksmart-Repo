import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';
import '../providers/ticket_providers.dart';

class TicketRaisingScreen extends ConsumerStatefulWidget {
  const TicketRaisingScreen({super.key});

  @override
  ConsumerState<TicketRaisingScreen> createState() =>
      _TicketRaisingScreenState();
}

class _TicketRaisingScreenState extends ConsumerState<TicketRaisingScreen> {
  final _formKey = GlobalKey<FormState>();
  final _titleController = TextEditingController();
  final _descriptionController = TextEditingController();
  final _stationIdController = TextEditingController();

  String _selectedPriority = 'Medium';
  String _selectedCategory = 'Hardware';

  final List<String> _priorities = ['Low', 'Medium', 'High', 'Critical'];
  final List<String> _categories = [
    'Hardware',
    'Software',
    'Connectivity',
    'Battery',
    'Safety',
    'Environmental',
    'Other'
  ];

  @override
  void dispose() {
    _titleController.dispose();
    _descriptionController.dispose();
    _stationIdController.dispose();
    super.dispose();
  }

  void _handleSubmit() {
    if (_formKey.currentState!.validate()) {
      final ticketData = {
        'title': _titleController.text,
        'description': _descriptionController.text,
        'stationId': _stationIdController.text,
        'priority': _selectedPriority,
        'category': _selectedCategory,
        'timestamp': DateTime.now().toIso8601String(),
      };

      ref.read(raiseTicketProvider.notifier).raiseTicket(ticketData);
    }
  }

  @override
  Widget build(BuildContext context) {
    ref.listen<AsyncValue<dynamic>>(raiseTicketProvider, (previous, next) {
      next.when(
        data: (data) {
          if (data != null && data.success) {
            // Show success message
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
                            data.message ?? 'Ticket raised successfully',
                            style: const TextStyle(fontSize: 14),
                          ),
                          if (data.ticketId != null)
                            Text(
                              'Ticket ID: ${data.ticketId}',
                              style: const TextStyle(
                                fontSize: 12,
                                fontStyle: FontStyle.italic,
                              ),
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
            // Clear form
            _titleController.clear();
            _descriptionController.clear();
            _stationIdController.clear();
            setState(() {
              _selectedPriority = 'Medium';
              _selectedCategory = 'Hardware';
            });
            ref.read(raiseTicketProvider.notifier).reset();
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
                          'Failed to raise ticket: $error',
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

    final isLoading = ref.watch(raiseTicketProvider).isLoading;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Raise Fault Ticket'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header Info
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: AppTheme.infoBlue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.info_outline,
                        color: AppTheme.infoBlue,
                        size: 32,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Text(
                          'Manually report a fault or issue that requires attention. All fields are required.',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 24),

                // Station ID
                Text(
                  'Station ID',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                TextFormField(
                  controller: _stationIdController,
                  decoration: const InputDecoration(
                    hintText: 'e.g., ST_101',
                    prefixIcon: Icon(Icons.location_on_outlined),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Please enter station ID';
                    }
                    return null;
                  },
                ),

                const SizedBox(height: 24),

                // Title
                Text(
                  'Issue Title',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                TextFormField(
                  controller: _titleController,
                  decoration: const InputDecoration(
                    hintText: 'Brief description of the issue',
                    prefixIcon: Icon(Icons.title),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Please enter a title';
                    }
                    return null;
                  },
                ),

                const SizedBox(height: 24),

                // Category
                Text(
                  'Category',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                DropdownButtonFormField<String>(
                  value: _selectedCategory,
                  decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.category_outlined),
                  ),
                  items: _categories.map((category) {
                    return DropdownMenuItem(
                      value: category,
                      child: Text(category),
                    );
                  }).toList(),
                  onChanged: (value) {
                    if (value != null) {
                      setState(() {
                        _selectedCategory = value;
                      });
                    }
                  },
                ),

                const SizedBox(height: 24),

                // Priority
                Text(
                  'Priority',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                DropdownButtonFormField<String>(
                  value: _selectedPriority,
                  decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.priority_high),
                  ),
                  items: _priorities.map((priority) {
                    return DropdownMenuItem(
                      value: priority,
                      child: Row(
                        children: [
                          Container(
                            width: 12,
                            height: 12,
                            decoration: BoxDecoration(
                              color: _getPriorityColor(priority),
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Text(priority),
                        ],
                      ),
                    );
                  }).toList(),
                  onChanged: (value) {
                    if (value != null) {
                      setState(() {
                        _selectedPriority = value;
                      });
                    }
                  },
                ),

                const SizedBox(height: 24),

                // Description
                Text(
                  'Detailed Description',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                TextFormField(
                  controller: _descriptionController,
                  maxLines: 5,
                  decoration: const InputDecoration(
                    hintText: 'Provide detailed information about the issue...',
                    alignLabelWithHint: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Please enter a description';
                    }
                    if (value.length < 10) {
                      return 'Description must be at least 10 characters';
                    }
                    return null;
                  },
                ),

                const SizedBox(height: 32),

                // Submit Button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: isLoading ? null : _handleSubmit,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.gradientStart,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    child: isLoading
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor:
                                  AlwaysStoppedAnimation<Color>(Colors.white),
                            ),
                          )
                        : const Text(
                            'Raise Ticket',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                  ),
                ),

                const SizedBox(height: 16),

                // Cancel Button
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton(
                    onPressed:
                        isLoading ? null : () => Navigator.of(context).pop(),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: AppTheme.gradientStart),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    child: const Text(
                      'Cancel',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.gradientStart,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _getPriorityColor(String priority) {
    switch (priority) {
      case 'Critical':
        return AppTheme.criticalRed;
      case 'High':
        return Colors.orange;
      case 'Medium':
        return AppTheme.warningYellow;
      case 'Low':
        return AppTheme.successGreen;
      default:
        return Colors.grey;
    }
  }
}
