# NAVSWAP Quick Start Guide

## 📦 Extract & Setup (5 minutes)

### 1. Extract the Archive
```bash
tar -xzf navswap_app.tar.gz
cd navswap_app
```

### 2. Install Dependencies
```bash
flutter pub get
```

### 3. Run the App
```bash
flutter run
```

That's it! The app will launch on your connected device/emulator.

## 🎯 What You Get

### ✅ Fully Functional Screens
1. **Authentication Flow**
   - Splash screen
   - Login/Signup
   - OTP verification
   - Role selection

2. **Customer App**
   - Home with AI recommendation card
   - Nearby stations list
   - Quick actions
   - Navigation ready

3. **Transporter App**
   - Dashboard with stats
   - Task management ready
   - Bottom navigation

### 📱 UI Features
- Modern, clean design matching uploaded image
- Gradient cards with shadows
- Smooth animations
- Bottom navigation with active states
- Responsive layouts

## 🔧 Quick Customization

### Change Colors
Edit `lib/core/theme/app_theme.dart`:
```dart
static const Color primaryColor = Color(0xFF1A1A1A);  // Change this
static const Color accentOrange = Color(0xFFFF6B4A);  // Change this
```

### Update API Base URL
Edit `lib/core/api/api_client.dart`:
```dart
static const String baseUrl = 'https://your-api.com/v1';
```

### Modify App Name
Edit `pubspec.yaml`:
```yaml
name: your_app_name
description: Your app description
```

## 📝 Next Steps

1. **Connect Backend**
   - Update API endpoints in `api_client.dart`
   - Implement actual authentication
   - Connect to real database

2. **Complete Remaining Screens**
   - Station detail with maps
   - Queue management with live updates
   - AI chat interface
   - Task detail for transporters

3. **Add Firebase**
   - Add `google-services.json` (Android)
   - Add `GoogleService-Info.plist` (iOS)
   - Configure push notifications

4. **Test & Deploy**
   - Run tests: `flutter test`
   - Build APK: `flutter build apk`
   - Submit to stores

## 🎨 Design System

### Colors
- **Primary**: #1A1A1A (Dark)
- **Accent Orange**: #FF6B4A
- **Accent Blue**: #4A90E2
- **Background**: #F8F9FA
- **Success**: #4CAF50
- **Error**: #FF5252

### Typography
- **Font**: Inter
- **Sizes**: 32/28/24/20/18/16/14/12

### Spacing
- **Small**: 8px
- **Medium**: 16px
- **Large**: 24px
- **XLarge**: 32px

## 🔑 Key Files to Know

| File | Purpose |
|------|---------|
| `main.dart` | App entry point |
| `app_router.dart` | Navigation configuration |
| `app_theme.dart` | Design system |
| `auth_service.dart` | Authentication logic |
| `api_client.dart` | API communication |
| `customer_home_screen.dart` | Main customer screen |
| `transporter_dashboard_screen.dart` | Main transporter screen |

## 💡 Tips

1. **Hot Reload**: Press `r` in terminal while app is running
2. **Debug Mode**: Use VS Code or Android Studio debugger
3. **State Management**: All state is managed with Riverpod
4. **Code Generation**: Run `flutter pub run build_runner build` after model changes

## 📚 Documentation

- `README.md` - Complete documentation
- `ARCHITECTURE.md` - Technical architecture
- `SETUP_GUIDE.md` - Detailed setup instructions
- `PROJECT_OVERVIEW.md` - Feature overview

## ⚠️ Requirements

- Flutter SDK >=3.0.0
- Dart SDK >=3.0.0
- Android Studio or VS Code
- Android SDK / Xcode

## 🐛 Common Issues

**Build fails?**
```bash
flutter clean
flutter pub get
flutter run
```

**Dependencies error?**
```bash
flutter pub upgrade
```

**iOS build fails?**
```bash
cd ios
pod install
cd ..
flutter run
```

## 📞 Need Help?

- Check full documentation in `README.md`
- Review `SETUP_GUIDE.md` for detailed steps
- Email: dev@navswap.com

---

**Ready to build the future of EV battery swaps!** 🚗⚡
