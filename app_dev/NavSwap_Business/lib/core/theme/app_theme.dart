import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Color Palette - Professional Black/White Theme
  static const Color darkBackground = Color(0xFF000000); // Pure Black
  static const Color cardBackground = Color(0xFF121212); // Dark Grey
  static const Color surfaceColor = Color(0xFF1E1E1E); // Lighter Grey

  // Gradient Colors
  static const Color gradientStart = Color(0xFFFFFFFF); // White
  static const Color gradientEnd = Color(0xFFB0B0B0); // Silver

  // Status Colors
  static const Color successGreen = Color(0xFF4CAF50); // Keep for semantics
  static const Color warningYellow = Color(0xFFFFB74D); // Keep for semantics
  static const Color criticalRed = Color(0xFFEF5350); // Keep for semantics
  static const Color infoBlue = Color(0xFF42A5F5); // Keep for semantics

  // Text Colors
  static const Color textPrimary = Color(0xFFFFFFFF);
  static const Color textSecondary = Color(0xFFB0B0B0); // Pure Grey
  static const Color textHint = Color(0xFF808080); // Grey

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: darkBackground,
      colorScheme: const ColorScheme.dark(
        primary: gradientStart, // White primary
        secondary: gradientEnd, // Silver secondary
        surface: cardBackground,
        background: darkBackground,
        error: criticalRed,
      ),
      textTheme: GoogleFonts.interTextTheme(
        const TextTheme(
          displayLarge: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.bold,
            color: textPrimary,
          ),
          displayMedium: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: textPrimary,
          ),
          displaySmall: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          headlineMedium: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          headlineSmall: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          titleLarge: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: textPrimary,
          ),
          titleMedium: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
            color: textPrimary,
          ),
          bodyLarge: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.normal,
            color: textPrimary,
          ),
          bodyMedium: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.normal,
            color: textSecondary,
          ),
          bodySmall: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.normal,
            color: textHint,
          ),
        ),
      ),
      cardTheme: CardThemeData(
        color: cardBackground,
        elevation: 4,
        shadowColor: Colors.white
            .withValues(alpha: 0.1), // Subtle white shadow for visibility
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: gradientStart, // White button
          foregroundColor: Colors.black, // Black text
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          elevation: 4,
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceColor,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide:
              const BorderSide(color: gradientStart, width: 2), // White border
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        hintStyle: const TextStyle(color: textHint),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: darkBackground,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: textPrimary,
        ),
      ),
    );
  }

  // Gradient Decorations
  static BoxDecoration gradientCard = BoxDecoration(
    gradient: const LinearGradient(
      colors: [Color.fromARGB(45, 255, 255, 255), gradientEnd],
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
    ),
    borderRadius: BorderRadius.circular(20),
    boxShadow: [
      BoxShadow(
        color: const Color.fromARGB(181, 255, 255, 255).withValues(alpha: 0.3),
        blurRadius: 16,
        offset: const Offset(0, 8),
      ),
    ],
  );

  static BoxDecoration darkCard = BoxDecoration(
    color: cardBackground,
    borderRadius: BorderRadius.circular(20),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withValues(alpha: 0.2),
        blurRadius: 12,
        offset: const Offset(0, 4),
      ),
    ],
  );
}
