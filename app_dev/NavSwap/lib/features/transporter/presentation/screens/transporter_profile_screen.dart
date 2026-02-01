import 'package:flutter/material.dart';

class TransporterProfileScreen extends StatelessWidget {
  const TransporterProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: const Center(child: Text('Transporter Profile Screen')),
    );
  }
}
