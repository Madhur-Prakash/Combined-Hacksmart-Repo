import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';

class LocationService {
  static Future<void> initialize() async {
    await Permission.location.request();
  }

  static Future<Position?> getCurrentLocation() async {
    final hasPermission = await Permission.location.isGranted;
    if (!hasPermission) return null;

    try {
      return await Geolocator.getCurrentPosition();
    } catch (e) {
      return null;
    }
  }
}
