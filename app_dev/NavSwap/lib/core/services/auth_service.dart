import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:jwt_decoder/jwt_decoder.dart';
import '../models/user_model.dart';
import '../api/api_client.dart';

final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService(ref);
});

class AuthService {
  final Ref _ref;
  final FlutterSecureStorage _storage = const FlutterSecureStorage();
  
  static const String _tokenKey = 'auth_token';
  static const String _userKey = 'user_data';
  
  UserModel? _currentUser;
  String? _token;
  
  AuthService(this._ref);
  
  bool get isAuthenticated => _token != null && !_isTokenExpired();
  UserModel? get currentUser => _currentUser;
  String? get token => _token;
  
  bool _isTokenExpired() {
    if (_token == null) return true;
    if (_token == 'mock_token_bypass_jwt') return false; // Bypass expiry for mock token
    try {
      return JwtDecoder.isExpired(_token!);
    } catch (e) {
      return true;
    }
  }
  
  Future<void> initialize() async {
    _token = await _storage.read(key: _tokenKey);
    final userData = await _storage.read(key: _userKey);
    
    if (userData != null) {
      try {
        _currentUser = UserModel.fromJson(jsonDecode(userData));
      } catch (e) {
        // Invalid user data, clear it
        await logout();
      }
    }
  }
  
  Future<bool> login({
    required String emailOrPhone,
    required String password,
    String? role,
  }) async {
    // Mock Login Bypass
    if (emailOrPhone == '9999999999' && password == '123456') {
      _token = 'mock_token_bypass_jwt';
      await _storage.write(key: _tokenKey, value: _token);
      
      _currentUser = UserModel(
        id: 'mock_user_id',
        name: 'Demo User',
        email: 'demo@navswap.com',
        phone: '9999999999',
        role: role ?? 'customer', // Use selected role or default to customer
        isVerified: true,
        isApproved: true,
        createdAt: DateTime.now(),
      );
      await _storage.write(key: _userKey, value: jsonEncode(_currentUser!.toJson()));
      
      return true;
    }

    try {
      final response = await _ref.read(apiClientProvider).login(
        emailOrPhone: emailOrPhone,
        password: password,
      );
      
      _token = response['token'] as String?;
      if (_token != null) {
        await _storage.write(key: _tokenKey, value: _token);
        _ref.read(apiClientProvider).setAuthToken(_token!);
      }
      
      // Store user data
      if (response['user'] != null) {
        _currentUser = UserModel.fromJson(response['user'] as Map<String, dynamic>);
        await _storage.write(key: _userKey, value: jsonEncode(_currentUser!.toJson()));
      }
      
      return true;
    } catch (e) {
      return false;
    }
  }
  
  Future<bool> signup({
    required String name,
    required String email,
    required String phone,
    required String password,
  }) async {
    try {
      final response = await _ref.read(apiClientProvider).signup(
        name: name,
        email: email,
        phone: phone,
        password: password,
      );
      
      return response['success'] == true;
    } catch (e) {
      return false;
    }
  }
  
  Future<bool> verifyOtp({
    required String phone,
    required String otp,
  }) async {
    // Mock OTP Bypass
    if (phone == '9999999999' && otp == '123456') {
      _token = 'mock_token_bypass_jwt';
      await _storage.write(key: _tokenKey, value: _token);
      
      _currentUser = UserModel(
        id: 'mock_user_id',
        name: 'Demo User',
        email: 'demo@navswap.com',
        phone: '9999999999',
        role: 'user',
        isVerified: true,
        isApproved: true,
        createdAt: DateTime.now(),
      );
      await _storage.write(key: _userKey, value: jsonEncode(_currentUser!.toJson()));
      
      return true;
    }

    try {
      final response = await _ref.read(apiClientProvider).verifyOtp(
        phone: phone,
        otp: otp,
      );
      
      _token = response['token'] as String?;
      if (_token != null) {
        await _storage.write(key: _tokenKey, value: _token);
        _ref.read(apiClientProvider).setAuthToken(_token!);
      }
      
      return true;
    } catch (e) {
      return false;
    }
  }
  
  Future<bool> selectRole(String role) async {
    try {
      final response = await _ref.read(apiClientProvider).updateRole(role);
      
      // Update user data
      if (response['user'] != null) {
        _currentUser = UserModel.fromJson(response['user'] as Map<String, dynamic>);
        await _storage.write(key: _userKey, value: jsonEncode(_currentUser!.toJson()));
      }
      
      return response['success'] == true;
    } catch (e) {
      return false;
    }
  }
  
  Future<void> logout() async {
    _token = null;
    _currentUser = null;
    await _storage.delete(key: _tokenKey);
    await _storage.delete(key: _userKey);
  }
}
