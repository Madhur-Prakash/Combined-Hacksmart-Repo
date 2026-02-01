import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../features/auth/presentation/screens/splash_screen.dart';
import '../../features/auth/presentation/screens/login_screen.dart';
import '../../features/auth/presentation/screens/signup_screen.dart';
import '../../features/auth/presentation/screens/otp_screen.dart';
import '../../features/auth/presentation/screens/role_selection_screen.dart';
import '../../features/customer/presentation/screens/customer_home_screen.dart';
import '../../features/customer/presentation/screens/station_detail_screen.dart';
import '../../features/customer/presentation/screens/queue_screen.dart';
import '../../features/customer/presentation/screens/history_screen.dart';
import '../../features/customer/presentation/screens/ai_chat_screen.dart';
import '../../features/transporter/presentation/screens/transporter_dashboard_screen.dart';
import '../../features/transporter/presentation/screens/task_detail_screen.dart';
import '../../features/transporter/presentation/screens/transporter_history_screen.dart';
import '../../features/transporter/presentation/screens/transporter_profile_screen.dart';
import '../../features/transporter/presentation/screens/transporter_verification_screen.dart';
import '../services/auth_service.dart';

final routerProvider = Provider<GoRouter>((ref) {
  final authService = ref.watch(authServiceProvider);

  return GoRouter(
    initialLocation: '/splash',
    debugLogDiagnostics: true,
    redirect: (context, state) {
      final isAuthenticated = authService.isAuthenticated;
      final isAuthRoute = state.matchedLocation.startsWith('/auth') ||
          state.matchedLocation == '/splash';

      if (!isAuthenticated && !isAuthRoute) {
        return '/auth/role-select';
      }

      if (isAuthenticated &&
          isAuthRoute &&
          state.matchedLocation != '/splash') {
        final userRole = authService.currentUser?.role;
        if (userRole == 'customer') {
          return '/customer/home';
        } else if (userRole == 'transporter') {
          return '/transporter/dashboard';
        }
      }

      return null;
    },
    routes: [
      GoRoute(
        path: '/splash',
        name: 'splash',
        builder: (context, state) => const SplashScreen(),
      ),

      // Auth Routes
      GoRoute(
        path: '/auth/login',
        name: 'login',
        builder: (context, state) {
          final role = state.uri.queryParameters['role'];
          return LoginScreen(role: role);
        },
      ),
      GoRoute(
        path: '/auth/signup',
        name: 'signup',
        builder: (context, state) => const SignupScreen(),
      ),
      GoRoute(
        path: '/auth/otp',
        name: 'otp',
        builder: (context, state) {
          final phone = state.uri.queryParameters['phone'] ?? '';
          return OtpScreen(phoneNumber: phone);
        },
      ),
      GoRoute(
        path: '/auth/role-select',
        name: 'roleSelect',
        builder: (context, state) => const RoleSelectionScreen(),
      ),

      // Customer Routes
      GoRoute(
        path: '/customer/home',
        name: 'customerHome',
        builder: (context, state) => const CustomerHomeScreen(),
      ),
      GoRoute(
        path: '/customer/station/:id',
        name: 'stationDetail',
        builder: (context, state) {
          final stationId = state.pathParameters['id']!;
          return StationDetailScreen(stationId: stationId);
        },
      ),
      GoRoute(
        path: '/customer/queue',
        name: 'queue',
        builder: (context, state) => const QueueScreen(),
      ),
      GoRoute(
        path: '/customer/history',
        name: 'customerHistory',
        builder: (context, state) => const HistoryScreen(),
      ),
      GoRoute(
        path: '/customer/ai-chat',
        name: 'aiChat',
        builder: (context, state) => const AiChatScreen(),
      ),

      // Transporter Routes
      GoRoute(
        path: '/transporter/dashboard',
        name: 'transporterDashboard',
        builder: (context, state) => const TransporterDashboardScreen(),
      ),
      GoRoute(
        path: '/transporter/verification',
        name: 'transporterVerification',
        builder: (context, state) => const TransporterVerificationScreen(),
      ),
      GoRoute(
        path: '/transporter/task/:id',
        name: 'taskDetail',
        builder: (context, state) {
          final taskId = state.pathParameters['id']!;
          return TaskDetailScreen(taskId: taskId);
        },
      ),
      GoRoute(
        path: '/transporter/history',
        name: 'transporterHistory',
        builder: (context, state) => const TransporterHistoryScreen(),
      ),
      GoRoute(
        path: '/transporter/profile',
        name: 'transporterProfile',
        builder: (context, state) => const TransporterProfileScreen(),
      ),
    ],
    errorBuilder: (context, state) => Scaffold(
      body: Center(
        child: Text('Error: ${state.error}'),
      ),
    ),
  );
});
