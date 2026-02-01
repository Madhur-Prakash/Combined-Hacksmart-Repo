# NAVSWAP BUSINESS - Station Staff Operations App

A professional Flutter mobile application for EV battery swap station operations management, designed exclusively for station staff.

## 🎯 Overview

NAVSWAP BUSINESS is a comprehensive station control dashboard that serves as:
- Station control dashboard
- Inventory management tool
- Fault monitoring system
- Workforce performance tracker
- AI-assisted operations console
- Battery scanner tool

## ✨ Features

### 🔐 Authentication
- Secure phone-based login
- OTP verification
- Staff approval validation
- Station assignment verification

### 📊 Dashboard
- Real-time station overview
- Active swaps monitoring
- Queue status
- Battery availability
- Charger health tracking
- AI-powered recommendations
- Performance metrics

### 👥 Queue Management
- Live queue monitoring
- Swap progress tracking
- Wait time estimates
- Bay status overview

### 📦 Inventory Management
- Real-time battery tracking
- Charge status monitoring
- Faulty battery alerts
- Low stock warnings
- Charging progress

### ⚠️ Fault Monitoring
- Critical fault alerts
- Warning notifications
- AI-powered diagnostics
- Suggested actions
- Resolution tracking

### 🤖 AI Recommendations
- Preventive maintenance alerts
- Staff allocation suggestions
- Battery reorder notifications
- Performance optimization tips
- Risk warnings

### 📱 Battery Scanner
- QR code scanning
- Battery status verification
- Quick status updates
- Location assignment
- Health check reports

### 👤 Staff Profile
- Personal performance metrics
- Work history
- Shift information
- Efficiency scores

### 📈 Work History
- Swap records
- Battery operations log
- Fault resolution history
- Performance analytics

## 🛠️ Technology Stack

- **Framework**: Flutter (Latest Stable)
- **State Management**: Riverpod
- **Navigation**: GoRouter
- **UI**: Material Design 3
- **Fonts**: Google Fonts (Inter)
- **Architecture**: Clean Architecture

## 📱 Screens

1. **Login Screen** - Phone authentication
2. **OTP Screen** - Verification
3. **Dashboard** - Control center
4. **Queue Screen** - Queue monitoring
5. **Inventory Screen** - Battery management
6. **Faults Screen** - Fault tracking
7. **Actions Screen** - AI recommendations
8. **Scanner Screen** - QR scanning
9. **Profile Screen** - Staff profile
10. **History Screen** - Work logs

## 🎨 Design System

### Color Palette
- **Background**: Dark theme (#0A0E14)
- **Card Background**: #151B26
- **Surface**: #1E2530
- **Gradient**: #FF6B3D → #FF9068
- **Success**: #4CAF50
- **Warning**: #FFB74D
- **Critical**: #EF5350
- **Info**: #42A5F5

### Typography
- **Font Family**: Inter (via Google Fonts)
- **Display**: 32-24px, Bold
- **Headlines**: 20-18px, Semi-bold
- **Body**: 16-14px, Regular
- **Small**: 12px, Regular

### Components
- **Cards**: 20px border radius, elevated shadows
- **Buttons**: 16px border radius, gradient background
- **Input Fields**: 16px border radius, filled style
- **Icons**: 24-32px, contextual colors

## 🚀 Getting Started

### Prerequisites
- Flutter SDK (3.0.0 or higher)
- Dart SDK (3.0.0 or higher)
- Android Studio / VS Code
- Android SDK / Xcode (for iOS)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd navswap_business
```

2. **Install dependencies**
```bash
flutter pub get
```

3. **Run the app**
```bash
flutter run
```

### Build

**Android APK**
```bash
flutter build apk --release
```

**Android App Bundle**
```bash
flutter build appbundle --release
```

**iOS**
```bash
flutter build ios --release
```

## 📁 Project Structure

```
lib/
├── main.dart
├── core/
│   ├── router/
│   │   └── app_router.dart
│   └── theme/
│       └── app_theme.dart
└── features/
    ├── auth/
    │   └── presentation/
    │       └── screens/
    │           ├── login_screen.dart
    │           └── otp_screen.dart
    ├── dashboard/
    │   └── presentation/
    │       └── screens/
    │           └── dashboard_screen.dart
    ├── queue/
    │   └── presentation/
    │       └── screens/
    │           └── queue_screen.dart
    ├── inventory/
    │   └── presentation/
    │       └── screens/
    │           └── inventory_screen.dart
    ├── faults/
    │   └── presentation/
    │       └── screens/
    │           └── faults_screen.dart
    ├── actions/
    │   └── presentation/
    │       └── screens/
    │           └── actions_screen.dart
    ├── scanner/
    │   └── presentation/
    │       └── screens/
    │           └── scanner_screen.dart
    ├── profile/
    │   └── presentation/
    │       └── screens/
    │           └── profile_screen.dart
    └── history/
        └── presentation/
            └── screens/
                └── history_screen.dart
```

## 🔌 API Integration

The app is designed to integrate with backend services for:
- Authentication (JWT)
- Real-time station data (WebSockets)
- Battery inventory management
- Fault reporting and tracking
- AI recommendations
- Performance analytics

### API Endpoints (To be configured)
```dart
// lib/core/constants/api_constants.dart
class ApiConstants {
  static const String baseUrl = 'YOUR_API_BASE_URL';
  static const String authEndpoint = '/api/auth';
  static const String stationEndpoint = '/api/station';
  static const String inventoryEndpoint = '/api/inventory';
  static const String faultsEndpoint = '/api/faults';
  static const String aiEndpoint = '/api/ai-recommendations';
}
```

## 📲 QR Scanner Setup

For production use, update the scanner implementation:

1. Add camera permissions in `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

2. Add camera usage description in `Info.plist` (iOS):
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is required for scanning battery QR codes</string>
```

3. Implement actual scanning logic using the `mobile_scanner` package

## 🔒 Security

- Secure JWT token storage
- Biometric authentication support
- Encrypted local data
- HTTPS API communication
- Session timeout management

## 🧪 Testing

```bash
# Run unit tests
flutter test

# Run integration tests
flutter test integration_test

# Generate coverage report
flutter test --coverage
```

## 📝 License

This project is proprietary software developed for NAVSWAP station operations.

## 👥 Team

- **Developed for**: NAVSWAP Operations
- **Type**: Internal Business Application
- **Target Users**: Station Staff

## 📞 Support

For technical support or issues:
- Email: support@navswap.com
- Contact Station Manager
- Internal IT Helpdesk

## 🔄 Version History

### Version 1.0.0 (Current)
- Initial release
- Authentication system
- Dashboard with station overview
- Queue management
- Inventory tracking
- Fault monitoring
- AI recommendations
- Battery scanner
- Staff profile and history

## 🚀 Future Enhancements

- [ ] Push notifications for critical alerts
- [ ] Offline mode support
- [ ] Advanced analytics dashboard
- [ ] Voice commands integration
- [ ] AR-based battery placement guide
- [ ] Multi-language support
- [ ] Dark/Light theme toggle
- [ ] Export reports functionality
- [ ] Biometric authentication
- [ ] Integration with wearable devices

## 📋 Notes

- This is a staff-only application
- Requires active station assignment
- Real-time data updates via WebSocket
- AI recommendations powered by backend ML models
- Regular updates required for optimal performance

---

**Built with ❤️ for NAVSWAP Station Operations**
