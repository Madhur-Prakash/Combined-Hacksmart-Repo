# NAVSWAP Setup Guide

## Prerequisites

### Required Software
1. **Flutter SDK** (3.0.0 or higher)
   ```bash
   flutter --version
   ```

2. **Dart SDK** (3.0.0 or higher)
   ```bash
   dart --version
   ```

3. **Android Studio** OR **VS Code**
   - Android Studio: Install Android SDK and emulator
   - VS Code: Install Flutter and Dart extensions

4. **Xcode** (macOS only for iOS development)

5. **Firebase CLI** (for notifications)
   ```bash
   npm install -g firebase-tools
   firebase login
   ```

## Step-by-Step Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/navswap_app.git
cd navswap_app

# Get Flutter dependencies
flutter pub get

# Check for issues
flutter doctor
```

### 2. Firebase Configuration

#### Android Setup
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or use existing
3. Add Android app
   - Package name: `com.navswap.app` (or your package)
   - Download `google-services.json`
   - Place in `android/app/`

#### iOS Setup
1. Add iOS app in Firebase Console
   - Bundle ID: `com.navswap.app`
   - Download `GoogleService-Info.plist`
   - Place in `ios/Runner/`

2. Update `ios/Runner/Info.plist`:
```xml
<key>NSLocationWhenInUseUsageDescription</key>
<string>We need your location to find nearby stations</string>
<key>NSLocationAlwaysUsageDescription</key>
<string>We need your location to provide real-time updates</string>
```

### 3. Google Maps Setup

#### Get API Keys
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Maps SDK for Android and iOS
3. Create API credentials

#### Android Configuration
Add to `android/app/src/main/AndroidManifest.xml`:
```xml
<application>
    <meta-data
        android:name="com.google.android.geo.API_KEY"
        android:value="YOUR_ANDROID_API_KEY"/>
</application>
```

#### iOS Configuration
Add to `ios/Runner/AppDelegate.swift`:
```swift
import GoogleMaps

GMSServices.provideAPIKey("YOUR_IOS_API_KEY")
```

### 4. Environment Configuration

Create `.env` file in root:
```env
API_BASE_URL=https://api.navswap.com/v1
API_KEY=your_api_key_here
GOOGLE_MAPS_KEY=your_maps_key_here
```

### 5. Generate Code

Run build_runner to generate required files:
```bash
flutter pub run build_runner build --delete-conflicting-outputs
```

This generates:
- `*.g.dart` (JSON serialization)
- `*.freezed.dart` (Immutable models)
- Riverpod providers

### 6. Configure Permissions

#### Android (`android/app/src/main/AndroidManifest.xml`)
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION"/>
<uses-permission android:name="android.permission.CAMERA"/>
```

#### iOS (`ios/Runner/Info.plist`)
```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to scan QR codes</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>We need your location to find nearby stations</string>
```

### 7. Run the App

#### On Emulator
```bash
# Android
flutter run

# iOS (macOS only)
flutter run
```

#### On Physical Device
```bash
# List connected devices
flutter devices

# Run on specific device
flutter run -d <device-id>
```

## Development Workflow

### Hot Reload
- Press `r` in terminal for hot reload
- Press `R` for hot restart

### Code Generation Watch
Keep this running while developing:
```bash
flutter pub run build_runner watch
```

### Debugging
```bash
# Run in debug mode
flutter run --debug

# Run with verbose logging
flutter run -v

# Profile mode for performance testing
flutter run --profile
```

## Common Issues & Solutions

### Issue 1: Build Runner Fails
```bash
flutter pub run build_runner clean
flutter pub get
flutter pub run build_runner build --delete-conflicting-outputs
```

### Issue 2: Pod Install Fails (iOS)
```bash
cd ios
pod deintegrate
pod install
cd ..
flutter clean
flutter run
```

### Issue 3: Gradle Build Fails (Android)
```bash
cd android
./gradlew clean
cd ..
flutter clean
flutter run
```

### Issue 4: Google Maps Not Showing
- Verify API key is correct
- Check if billing is enabled in Google Cloud
- Ensure Maps SDK is enabled

## Building for Release

### Android APK
```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

### Android App Bundle (for Play Store)
```bash
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

### iOS IPA
```bash
flutter build ipa --release
# Output: build/ios/ipa/navswap_app.ipa
```

## Testing

### Run All Tests
```bash
flutter test
```

### Run Specific Test File
```bash
flutter test test/auth_test.dart
```

### Integration Tests
```bash
flutter drive --target=test_driver/app.dart
```

### Coverage Report
```bash
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html
open coverage/html/index.html
```

## Code Quality

### Analyze Code
```bash
flutter analyze
```

### Format Code
```bash
flutter format .
```

### Check Outdated Packages
```bash
flutter pub outdated
```

## Deployment Checklist

- [ ] Update version in `pubspec.yaml`
- [ ] Run all tests
- [ ] Test on physical devices
- [ ] Update Firebase config
- [ ] Configure release signing
- [ ] Generate release builds
- [ ] Test release builds
- [ ] Submit to stores

## Support

For issues or questions:
- GitHub Issues: [github.com/yourusername/navswap_app/issues]
- Email: dev@navswap.com
- Slack: #navswap-dev

## Resources

- [Flutter Docs](https://docs.flutter.dev/)
- [Riverpod Docs](https://riverpod.dev/)
- [Firebase Flutter](https://firebase.flutter.dev/)
- [GoRouter Guide](https://pub.dev/packages/go_router)
