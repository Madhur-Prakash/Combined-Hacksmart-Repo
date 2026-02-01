import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../../core/theme/app_theme.dart';

class ScannerScreen extends ConsumerStatefulWidget {
  const ScannerScreen({super.key});

  @override
  ConsumerState<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends ConsumerState<ScannerScreen> {
  bool _isScanning = true;
  String? _scannedBatteryId;

  void _simulateScan() {
    setState(() {
      _isScanning = false;
      _scannedBatteryId = 'BAT-1045';
    });
  }

  void _resetScanner() {
    setState(() {
      _isScanning = true;
      _scannedBatteryId = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Battery Scanner'),
        actions: [
          if (!_isScanning)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _resetScanner,
            ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            if (_isScanning) ...[
              Expanded(
                child: Stack(
                  children: [
                    // Scanner viewfinder
                    Container(
                      color: Colors.black,
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              width: 280,
                              height: 280,
                              decoration: BoxDecoration(
                                border: Border.all(
                                  color: AppTheme.gradientStart,
                                  width: 3,
                                ),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Stack(
                                children: [
                                  // Corner brackets
                                  Positioned(
                                    top: 0,
                                    left: 0,
                                    child: Container(
                                      width: 40,
                                      height: 40,
                                      decoration: const BoxDecoration(
                                        border: Border(
                                          top: BorderSide(color: AppTheme.gradientStart, width: 4),
                                          left: BorderSide(color: AppTheme.gradientStart, width: 4),
                                        ),
                                      ),
                                    ),
                                  ),
                                  Positioned(
                                    top: 0,
                                    right: 0,
                                    child: Container(
                                      width: 40,
                                      height: 40,
                                      decoration: const BoxDecoration(
                                        border: Border(
                                          top: BorderSide(color: AppTheme.gradientStart, width: 4),
                                          right: BorderSide(color: AppTheme.gradientStart, width: 4),
                                        ),
                                      ),
                                    ),
                                  ),
                                  Positioned(
                                    bottom: 0,
                                    left: 0,
                                    child: Container(
                                      width: 40,
                                      height: 40,
                                      decoration: const BoxDecoration(
                                        border: Border(
                                          bottom: BorderSide(color: AppTheme.gradientStart, width: 4),
                                          left: BorderSide(color: AppTheme.gradientStart, width: 4),
                                        ),
                                      ),
                                    ),
                                  ),
                                  Positioned(
                                    bottom: 0,
                                    right: 0,
                                    child: Container(
                                      width: 40,
                                      height: 40,
                                      decoration: const BoxDecoration(
                                        border: Border(
                                          bottom: BorderSide(color: AppTheme.gradientStart, width: 4),
                                          right: BorderSide(color: AppTheme.gradientStart, width: 4),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 32),
                            const Text(
                              'Align QR code within frame',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    
                    // Flash toggle
                    Positioned(
                      top: 20,
                      right: 20,
                      child: IconButton(
                        icon: const Icon(Icons.flash_off, color: Colors.white, size: 32),
                        onPressed: () {},
                      ),
                    ),
                  ],
                ),
              ),
              
              // Bottom instructions
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: AppTheme.cardBackground,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.qr_code_scanner, color: AppTheme.gradientStart, size: 48),
                    const SizedBox(height: 16),
                    Text(
                      'Scan Battery QR Code',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Position the QR code within the frame to scan',
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _simulateScan,
                        child: const Text('Simulate Scan (Demo)'),
                      ),
                    ),
                  ],
                ),
              ),
            ] else ...[
              // Scanned Result
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      // Success Animation
                      Container(
                        padding: const EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          color: AppTheme.successGreen.withOpacity(0.1),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.check_circle,
                          color: AppTheme.successGreen,
                          size: 80,
                        ),
                      ),
                      
                      const SizedBox(height: 24),
                      
                      Text(
                        'Scan Successful',
                        style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          color: AppTheme.successGreen,
                        ),
                      ),
                      
                      const SizedBox(height: 32),
                      
                      // Battery Info Card
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(24),
                        decoration: AppTheme.darkCard,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(16),
                                  decoration: BoxDecoration(
                                    color: AppTheme.gradientStart.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                  child: const Icon(
                                    Icons.battery_charging_full,
                                    color: AppTheme.gradientStart,
                                    size: 40,
                                  ),
                                ),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        _scannedBatteryId ?? '',
                                        style: Theme.of(context).textTheme.headlineMedium,
                                      ),
                                      const SizedBox(height: 4),
                                      Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                        decoration: BoxDecoration(
                                          color: AppTheme.successGreen.withOpacity(0.1),
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                        child: const Text(
                                          'READY',
                                          style: TextStyle(
                                            color: AppTheme.successGreen,
                                            fontSize: 12,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                            
                            const SizedBox(height: 24),
                            
                            _buildInfoRow(context, 'Charge Level', '100%'),
                            const Divider(height: 24, color: AppTheme.surfaceColor),
                            _buildInfoRow(context, 'Health Status', 'Excellent'),
                            const Divider(height: 24, color: AppTheme.surfaceColor),
                            _buildInfoRow(context, 'Location', 'Bay 7'),
                            const Divider(height: 24, color: AppTheme.surfaceColor),
                            _buildInfoRow(context, 'Last Swap', '2 hours ago'),
                            const Divider(height: 24, color: AppTheme.surfaceColor),
                            _buildInfoRow(context, 'Cycle Count', '245'),
                          ],
                        ),
                      ),
                      
                      const SizedBox(height: 24),
                      
                      Text(
                        'Update Battery Status',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      
                      const SizedBox(height: 16),
                      
                      // Status Update Buttons
                      GridView.count(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        crossAxisCount: 2,
                        mainAxisSpacing: 12,
                        crossAxisSpacing: 12,
                        childAspectRatio: 1.5,
                        children: [
                          _buildStatusButton(
                            context,
                            'Mark Charged',
                            Icons.battery_charging_full,
                            AppTheme.successGreen,
                          ),
                          _buildStatusButton(
                            context,
                            'In Use',
                            Icons.electric_car,
                            AppTheme.infoBlue,
                          ),
                          _buildStatusButton(
                            context,
                            'Report Faulty',
                            Icons.battery_alert,
                            AppTheme.criticalRed,
                          ),
                          _buildStatusButton(
                            context,
                            'Maintenance',
                            Icons.build,
                            AppTheme.warningYellow,
                          ),
                        ],
                      ),
                      
                      const SizedBox(height: 24),
                      
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _resetScanner,
                          child: const Text('Scan Another Battery'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(BuildContext context, String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.titleMedium,
        ),
      ],
    );
  }

  Widget _buildStatusButton(BuildContext context, String label, IconData icon, Color color) {
    return ElevatedButton(
      onPressed: () {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Battery status updated: $label'),
            backgroundColor: color,
          ),
        );
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: color.withOpacity(0.1),
        foregroundColor: color,
        padding: const EdgeInsets.all(16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: color),
        ),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 32),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
