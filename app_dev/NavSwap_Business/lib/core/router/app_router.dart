import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../features/auth/presentation/screens/login_screen.dart';
import '../../features/auth/presentation/screens/otp_screen.dart';
import '../../features/dashboard/presentation/screens/dashboard_screen.dart';
import '../../features/queue/presentation/screens/queue_screen.dart';
import '../../features/inventory/presentation/screens/inventory_screen.dart';
import '../../features/faults/presentation/screens/faults_screen.dart';
import '../../features/actions/presentation/screens/actions_screen.dart';
import '../../features/profile/presentation/screens/profile_screen.dart';
import '../../features/history/presentation/screens/history_screen.dart';
import '../../features/scanner/presentation/screens/qr_scanner_screen.dart';
import '../../features/scanner/presentation/screens/swap_details_screen.dart';
import '../../features/scanner/domain/models/verify_response_model.dart';
import '../../features/ticket/presentation/screens/ticket_raising_screen.dart';

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: '/login',
    routes: [
      GoRoute(
        path: '/login',
        builder: (context, state) => const LoginScreen(),
      ),
      GoRoute(
        path: '/otp',
        builder: (context, state) {
          final phone = state.extra as String?;
          return OtpScreen(phoneNumber: phone ?? '');
        },
      ),
      GoRoute(
        path: '/staff/dashboard',
        builder: (context, state) => const DashboardScreen(),
      ),
      GoRoute(
        path: '/staff/queue',
        builder: (context, state) => const QueueScreen(),
      ),
      GoRoute(
        path: '/staff/inventory',
        builder: (context, state) => const InventoryScreen(),
      ),
      GoRoute(
        path: '/staff/faults',
        builder: (context, state) => const FaultsScreen(),
      ),
      GoRoute(
        path: '/staff/ticket',
        builder: (context, state) => const TicketRaisingScreen(),
      ),
      GoRoute(
        path: '/staff/actions',
        builder: (context, state) => const ActionsScreen(),
      ),
      GoRoute(
        path: '/staff/scanner',
        builder: (context, state) => const QRScannerScreen(),
      ),
      GoRoute(
        path: '/staff/swap-details',
        builder: (context, state) {
          final verifyData = state.extra as VerifyResponseModel;
          return SwapDetailsScreen(verifyData: verifyData);
        },
      ),
      GoRoute(
        path: '/staff/profile',
        builder: (context, state) => const ProfileScreen(),
      ),
      GoRoute(
        path: '/staff/history',
        builder: (context, state) => const HistoryScreen(),
      ),
    ],
  );
});

final appRouter = GoRouter(
  routes: [
    // ...existing routes...

    GoRoute(
      path: '/staff/scanner',
      builder: (context, state) => const QRScannerScreen(),
    ),
    GoRoute(
      path: '/staff/swap-details',
      builder: (context, state) {
        final verifyData = state.extra as VerifyResponseModel;
        return SwapDetailsScreen(verifyData: verifyData);
      },
    ),
  ],
);
