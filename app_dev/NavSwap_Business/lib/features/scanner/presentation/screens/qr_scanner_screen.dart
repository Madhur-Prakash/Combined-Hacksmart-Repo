import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';
import '../providers/scanner_providers.dart';

class QRScannerScreen extends ConsumerStatefulWidget {
  const QRScannerScreen({super.key});

  @override
  ConsumerState<QRScannerScreen> createState() => _QRScannerScreenState();
}

class _QRScannerScreenState extends ConsumerState<QRScannerScreen> {
  MobileScannerController cameraController = MobileScannerController();
  bool _isProcessing = false;

  @override
  void dispose() {
    cameraController.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) async {
    if (_isProcessing) return;

    final List<Barcode> barcodes = capture.barcodes;
    if (barcodes.isEmpty) return;

    final String? qrCode = barcodes.first.rawValue;
    if (qrCode == null || qrCode.isEmpty) return;

    setState(() => _isProcessing = true);

    // Verify QR code
    await ref.read(verifyQRProvider.notifier).verifyQRCode(qrCode);

    final verifyState = ref.read(verifyQRProvider);

    verifyState.when(
      data: (verifyData) {
        if (verifyData != null) {
          // Navigate to swap details screen
          context.push('/staff/swap-details', extra: verifyData);
        } else {
          _showErrorDialog(
              'Invalid QR Code', 'The scanned QR code is not valid.');
        }
        setState(() => _isProcessing = false);
      },
      loading: () {},
      error: (error, _) {
        _showErrorDialog('Verification Failed', error.toString());
        setState(() => _isProcessing = false);
      },
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
    final verifyState = ref.watch(verifyQRProvider);

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Scan QR Code'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.pop(),
        ),
        actions: [
          IconButton(
            icon: Icon(
              cameraController.torchEnabled ? Icons.flash_on : Icons.flash_off,
            ),
            onPressed: () => cameraController.toggleTorch(),
          ),
          IconButton(
            icon: const Icon(Icons.flip_camera_ios),
            onPressed: () => cameraController.switchCamera(),
          ),
        ],
      ),
      body: Stack(
        children: [
          // Camera Preview
          MobileScanner(
            controller: cameraController,
            onDetect: _onDetect,
          ),

          // Overlay
          CustomPaint(
            painter: ScannerOverlayPainter(),
            child: Container(),
          ),

          // Instructions
          Positioned(
            bottom: 100,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(30),
                    ),
                    child: const Text(
                      'Align QR code within the frame',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  if (verifyState.isLoading || _isProcessing) ...[
                    const SizedBox(height: 20),
                    const CircularProgressIndicator(
                      color: AppTheme.gradientStart,
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ScannerOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final double scanAreaSize = size.width * 0.7;
    final double left = (size.width - scanAreaSize) / 2;
    final double top = (size.height - scanAreaSize) / 2;

    // Draw semi-transparent overlay
    final Paint overlayPaint = Paint()
      ..color = Colors.black.withOpacity(0.5)
      ..style = PaintingStyle.fill;

    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, top), overlayPaint);
    canvas.drawRect(Rect.fromLTWH(0, top, left, scanAreaSize), overlayPaint);
    canvas.drawRect(Rect.fromLTWH(left + scanAreaSize, top, left, scanAreaSize),
        overlayPaint);
    canvas.drawRect(
        Rect.fromLTWH(0, top + scanAreaSize, size.width,
            size.height - top - scanAreaSize),
        overlayPaint);

    // Draw corner brackets
    final Paint bracketPaint = Paint()
      ..color = AppTheme.gradientStart
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;

    const double cornerLength = 30;

    // Top-left corner
    canvas.drawLine(
        Offset(left, top), Offset(left + cornerLength, top), bracketPaint);
    canvas.drawLine(
        Offset(left, top), Offset(left, top + cornerLength), bracketPaint);

    // Top-right corner
    canvas.drawLine(Offset(left + scanAreaSize, top),
        Offset(left + scanAreaSize - cornerLength, top), bracketPaint);
    canvas.drawLine(Offset(left + scanAreaSize, top),
        Offset(left + scanAreaSize, top + cornerLength), bracketPaint);

    // Bottom-left corner
    canvas.drawLine(Offset(left, top + scanAreaSize),
        Offset(left + cornerLength, top + scanAreaSize), bracketPaint);
    canvas.drawLine(Offset(left, top + scanAreaSize),
        Offset(left, top + scanAreaSize - cornerLength), bracketPaint);

    // Bottom-right corner
    canvas.drawLine(
        Offset(left + scanAreaSize, top + scanAreaSize),
        Offset(left + scanAreaSize - cornerLength, top + scanAreaSize),
        bracketPaint);
    canvas.drawLine(
        Offset(left + scanAreaSize, top + scanAreaSize),
        Offset(left + scanAreaSize, top + scanAreaSize - cornerLength),
        bracketPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
