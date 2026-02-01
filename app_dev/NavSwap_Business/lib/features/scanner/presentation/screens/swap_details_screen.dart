import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';
import '../../domain/models/verify_response_model.dart';
import '../providers/scanner_providers.dart';

class SwapDetailsScreen extends ConsumerStatefulWidget {
  final VerifyResponseModel verifyData;

  const SwapDetailsScreen({
    super.key,
    required this.verifyData,
  });

  @override
  ConsumerState<SwapDetailsScreen> createState() => _SwapDetailsScreenState();
}

class _SwapDetailsScreenState extends ConsumerState<SwapDetailsScreen> {
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    // Reset swap state when screen loads
    Future.microtask(() => ref.read(completeSwapProvider.notifier).reset());
  }

  void _confirmSwap() async {
    setState(() => _isProcessing = true);

    await ref.read(completeSwapProvider.notifier).completeSwap(
          userId: widget.verifyData.userId,
          currentBatteryId: widget.verifyData.currentBatteryId,
          assignedBatteryId: widget.verifyData.assignedBatteryId,
        );

    final swapState = ref.read(completeSwapProvider);

    swapState.when(
      data: (swapData) {
        if (swapData != null && swapData.success) {
          _showSuccessDialog();
        } else {
          _showErrorDialog(
              'Swap Failed', 'Failed to complete the battery swap.');
          setState(() => _isProcessing = false);
        }
      },
      loading: () {},
      error: (error, _) {
        _showErrorDialog('Error', error.toString());
        setState(() => _isProcessing = false);
      },
    );
  }

  void _showSuccessDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Column(
          children: [
            Icon(Icons.check_circle, color: AppTheme.successGreen, size: 64),
            SizedBox(height: 16),
            Text('Swap Completed!', textAlign: TextAlign.center),
          ],
        ),
        content: const Text(
          'Battery swap has been successfully completed.',
          textAlign: TextAlign.center,
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context); // Close dialog
              context.go('/staff/scanner'); // Return to scanner screen
            },
            child: const Text('Continue Scanning'),
          ),
        ],
      ),
    );
  }

  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Battery Swap Details'),
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Customer Information Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: AppTheme.darkCard,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.gradientStart.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.person,
                          color: AppTheme.gradientStart,
                          size: 32,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Customer Information',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleSmall
                                  ?.copyWith(
                                    color: Colors.grey,
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              widget.verifyData.userName,
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Divider(color: Colors.grey.withOpacity(0.3)),
                  const SizedBox(height: 16),
                  _buildInfoRow('User ID', widget.verifyData.userId),
                  const SizedBox(height: 12),
                  _buildInfoRow(
                      'Queue Position', widget.verifyData.queuePosition),
                  const SizedBox(height: 12),
                  _buildInfoRow(
                      'Wait Time', widget.verifyData.estimatedWaitTime),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Vehicle Information Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: AppTheme.darkCard,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.infoBlue.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.electric_car,
                          color: AppTheme.infoBlue,
                          size: 32,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Vehicle Information',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleSmall
                                  ?.copyWith(
                                    color: Colors.grey,
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              widget.verifyData.vehicleType,
                              style: Theme.of(context).textTheme.titleLarge,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Divider(color: Colors.grey.withOpacity(0.3)),
                  const SizedBox(height: 16),
                  _buildInfoRow(
                      'Vehicle Number', widget.verifyData.vehicleNumber),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Battery Swap Information
            Container(
              padding: const EdgeInsets.all(20),
              decoration: AppTheme.darkCard,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Battery Swap Details',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 20),

                  // Current Battery
                  Row(
                    children: [
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: AppTheme.warningYellow.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: AppTheme.warningYellow.withOpacity(0.3),
                            ),
                          ),
                          child: Column(
                            children: [
                              const Icon(
                                Icons.battery_2_bar,
                                color: AppTheme.warningYellow,
                                size: 48,
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                'Current Battery',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                widget.verifyData.currentBatteryId,
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                '${widget.verifyData.currentBatteryLevel}%',
                                style: const TextStyle(
                                  color: AppTheme.warningYellow,
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 12),
                        child: Icon(
                          Icons.swap_horiz,
                          size: 32,
                          color: AppTheme.gradientStart,
                        ),
                      ),
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: AppTheme.successGreen.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: AppTheme.successGreen.withOpacity(0.3),
                            ),
                          ),
                          child: Column(
                            children: [
                              const Icon(
                                Icons.battery_full,
                                color: AppTheme.successGreen,
                                size: 48,
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                'New Battery',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                widget.verifyData.assignedBatteryId,
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                '${widget.verifyData.assignedBatteryLevel}%',
                                style: const TextStyle(
                                  color: AppTheme.successGreen,
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 32),

            // Confirm Button
            SizedBox(
              width: double.infinity,
              height: 56,
              child: ElevatedButton(
                onPressed: _isProcessing ? null : _confirmSwap,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.successGreen,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isProcessing
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: 2,
                        ),
                      )
                    : const Text(
                        'Confirm Battery Swap',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
              ),
            ),

            const SizedBox(height: 16),

            // Cancel Button
            SizedBox(
              width: double.infinity,
              height: 56,
              child: OutlinedButton(
                onPressed: _isProcessing ? null : () => context.pop(),
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: Colors.grey),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Cancel',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Colors.grey,
            fontSize: 14,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
      ],
    );
  }
}
