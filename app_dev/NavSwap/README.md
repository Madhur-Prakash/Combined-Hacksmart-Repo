# NAVSWAP - AI-Powered EV Battery Swap Ecosystem

A production-grade Flutter mobile application for an AI-powered EV battery swap ecosystem serving both customers and transporters.

## 🎯 Features

### Customer App
- **AI-Powered Recommendations**: Smart station suggestions based on location, traffic, and availability
- **Smart Home Dashboard**: Hero AI card with best station recommendations
- **Station Discovery**: Find nearby stations with real-time data
- **Queue Management**: Reserve slots and track position in real-time
- **AI Assistant**: Chat interface for station comparisons and recommendations
- **History & Analytics**: Track swaps, time saved, and CO2 reduction

### Transporter App
- **Flash Job Requests**: 10-second animated task acceptance system
- **Task Management**: Pickup, transit, and drop confirmation workflow
- **Earnings Dashboard**: Real-time credits, completion rates, and efficiency scores
- **Route Optimization**: AI-powered route suggestions
- **Performance Metrics**: Track deliveries, on-time percentage, and tier badges

## 🏗️ Architecture

### Tech Stack
- **Framework**: Flutter 3.x
- **State Management**: Riverpod
- **Navigation**: GoRouter
- **Networking**: Dio + Retrofit
- **Local Storage**: Hive + Flutter Secure Storage
- **Maps**: Google Maps Flutter
- **Notifications**: Firebase Cloud Messaging
- **Architecture**: Clean Architecture with Feature-First Structure

### Project Structure
```
lib/
├── core/
│   ├── api/                 # API clients and endpoints
│   ├── models/              # Shared data models
│   ├── router/              # GoRouter configuration
│   ├── services/            # Core services (auth, location, notifications)
│   └── theme/               # App theme and styling
├── features/
│   ├── auth/                # Authentication flows
│   │   └── presentation/
│   │       └── screens/
│   ├── customer/            # Customer-specific features
│   │   └── presentation/
│   │       └── screens/
│   └── transporter/         # Transporter-specific features
│       └── presentation/
│           └── screens/
└── main.dart
```

## 🚀 Getting Started

### Prerequisites
- Flutter SDK >=3.0.0
- Dart SDK >=3.0.0
- Android Studio / VS Code
- Firebase project (for notifications)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/navswap_app.git
   cd navswap_app
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Generate code**
   ```bash
   flutter pub run build_runner build --delete-conflicting-outputs
   ```

4. **Configure Firebase**
   - Add `google-services.json` to `android/app/`
   - Add `GoogleService-Info.plist` to `ios/Runner/`

5. **Run the app**
   ```bash
   flutter run
   ```

## 🔐 Authentication Flow

1. **Splash Screen** → Token check & auto-login
2. **Login/Signup** → Phone/Email + Password
3. **OTP Verification** → Mandatory for both roles
4. **Role Selection** → Choose Customer or Transporter
5. **Transporter Verification** → ID/License upload (if transporter)
6. **Dashboard** → Navigate to role-specific home

## 🎨 UI/UX Highlights

- **Modern Design**: Clean, minimalist interface inspired by top logistics apps
- **Gradient Cards**: Eye-catching AI recommendation cards
- **Smooth Animations**: Lottie animations for loading states
- **Bottom Navigation**: Context-aware navigation with active state indicators
- **Real-time Updates**: WebSocket integration for live data
- **Offline Support**: Cached data for seamless offline experience

## 🧠 AI Integration

The app integrates with backend AI services for:
- **Station Recommendations**: ML-based scoring considering multiple factors
- **Wait Time Predictions**: Historical data + real-time traffic analysis
- **Route Optimization**: Shortest path with traffic consideration
- **Demand Forecasting**: Predict surge areas for transporters
- **Conversational AI**: Natural language queries about stations and system

## 📱 Key Screens

### Customer
1. **Home**: AI recommendation card + nearby stations list
2. **Station Detail**: Queue forecast, reliability metrics, charger status
3. **Queue Status**: Live position tracking with QR code
4. **AI Chat**: Conversational assistant for recommendations
5. **History**: Personal dashboard with analytics

### Transporter
1. **Dashboard**: Credits, deliveries, efficiency score
2. **Flash Job Request**: Full-screen overlay with countdown
3. **Task Detail**: Pickup → Transit → Drop workflow
4. **History**: Earnings and completion statistics
5. **Profile**: Performance metrics and tier badge

## 🔄 State Management

Using **Riverpod** for:
- Authentication state
- User profile management
- Station data caching
- Queue status updates
- Task management
- Real-time notifications

## 🌐 API Integration

### Endpoints
```dart
POST   /auth/login              # User login
POST   /auth/signup             # User registration
POST   /auth/verify-otp         # OTP verification
POST   /auth/role               # Set user role
GET    /stations/nearby         # Get nearby stations
GET    /stations/:id            # Station details
POST   /ai/recommend-station    # AI recommendation
POST   /queue/join              # Join queue
GET    /queue/:id               # Queue status
GET    /tasks/available         # Available tasks
POST   /tasks/:id/accept        # Accept task
PUT    /tasks/:id/status        # Update task status
POST   /ai/chat                 # AI chat message
```

## 🔔 Notifications

Push notifications for:
- Better station available nearby
- Queue delays or position updates
- Swap ready confirmation
- Task assignment (transporters)
- System alerts and maintenance

## 🛠️ Development

### Code Generation
```bash
# Run build_runner for freezed, json_serializable, riverpod
flutter pub run build_runner watch
```

### Testing
```bash
# Run unit tests
flutter test

# Run integration tests
flutter drive --target=test_driver/app.dart
```

### Building
```bash
# Android APK
flutter build apk --release

# iOS IPA
flutter build ipa --release
```

## 📦 Dependencies

Key packages:
- `flutter_riverpod` - State management
- `go_router` - Navigation
- `dio` - HTTP client
- `hive` - Local database
- `google_maps_flutter` - Maps integration
- `firebase_messaging` - Push notifications
- `qr_flutter` - QR code generation
- `fl_chart` - Data visualization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- **Product Design**: Modern logistics-inspired UI
- **Backend Integration**: REST + WebSocket APIs
- **AI/ML**: Station recommendation engine
- **Mobile Development**: Flutter + Clean Architecture

## 🔮 Future Enhancements

- [ ] Dark mode support
- [ ] Multi-language support
- [ ] Apple Pay / Google Pay integration
- [ ] Augmented reality station finder
- [ ] Voice commands for AI assistant
- [ ] Offline queue management
- [ ] Advanced analytics dashboard
- [ ] Social features (reviews, ratings)

## 📞 Support

For support, email support@navswap.com or join our Slack channel.

---

Built with ❤️ using Flutter
