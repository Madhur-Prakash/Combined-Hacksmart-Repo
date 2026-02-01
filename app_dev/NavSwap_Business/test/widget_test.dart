import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../lib/main.dart';

void main() {
  testWidgets('App renders smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ProviderScope(child: NavSwapBusinessApp()));

    // Verify the app starts (at login screen)
    await tester.pumpAndSettle();
    expect(find.text('NAVSWAP'), findsOneWidget);
  });
}
