import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../../core/theme/app_theme.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _phoneController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;

  @override
  void dispose() {
    _phoneController.dispose();
    super.dispose();
  }

  void _handleLogin() {
    if (_formKey.currentState!.validate()) {
      setState(() => _isLoading = true);

      // Simulate API call
      Future.delayed(const Duration(seconds: 2), () {
        setState(() => _isLoading = false);
        context.push('/otp', extra: _phoneController.text);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 40),

                // Logo/Header Section
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.gradientCard,
                  child: Row(
                    children: [
                      Container(
                        width: 60,
                        height: 60,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Image.asset(
                          'assets/icons/LogoBackgroundRemoved.png',
                          color: Colors.white,
                          fit: BoxFit.contain,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'NAVSWAP',
                              style: Theme.of(context)
                                  .textTheme
                                  .headlineMedium
                                  ?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                            Text(
                              'Station Staff Operations',
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(
                                    color: Colors.white.withOpacity(0.9),
                                  ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 48),

                Text(
                  'Staff Login',
                  style: Theme.of(context).textTheme.displaySmall,
                ),

                const SizedBox(height: 8),

                Text(
                  'Enter your registered mobile number to continue',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),

                const SizedBox(height: 32),

                // Phone Number Input
                TextFormField(
                  controller: _phoneController,
                  keyboardType: TextInputType.phone,
                  style: const TextStyle(color: AppTheme.textPrimary),
                  decoration: const InputDecoration(
                    labelText: 'Mobile Number',
                    hintText: 'Enter 10-digit mobile number',
                    prefixIcon: Icon(Icons.phone_outlined,
                        color: AppTheme.textSecondary),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Please enter mobile number';
                    }
                    if (value.length != 10) {
                      return 'Please enter valid 10-digit number';
                    }
                    return null;
                  },
                ),

                const SizedBox(height: 32),

                // Login Button
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : _handleLogin,
                    child: _isLoading
                        ? const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2,
                            ),
                          )
                        : const Text('Get OTP'),
                  ),
                ),

                const SizedBox(height: 24),

                // Info Cards
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: AppTheme.darkCard,
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppTheme.infoBlue.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.info_outline,
                          color: AppTheme.infoBlue,
                          size: 24,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Staff Access Only',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'This app is exclusively for authorized station personnel',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 48),

                Center(
                  child: Column(
                    children: [
                      Text(
                        'Need Help?',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      const SizedBox(height: 8),
                      TextButton(
                        onPressed: () {},
                        child: const Text('Contact Station Manager'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
