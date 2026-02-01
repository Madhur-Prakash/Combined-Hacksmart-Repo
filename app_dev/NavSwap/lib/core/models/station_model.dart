class StationModel {
  final String id;
  final String name;
  final double latitude;
  final double longitude;
  final double distance;
  final int waitTime;
  final double aiScore;
  final double reliability;
  final int availableSlots;
  final int totalSlots;
  final String status;
  final List<String> amenities;
  final String? imageUrl;

  StationModel({
    required this.id,
    required this.name,
    required this.latitude,
    required this.longitude,
    required this.distance,
    required this.waitTime,
    required this.aiScore,
    required this.reliability,
    required this.availableSlots,
    required this.totalSlots,
    required this.status,
    this.amenities = const [],
    this.imageUrl,
  });

  factory StationModel.fromJson(Map<String, dynamic> json) {
    return StationModel(
      id: json['id'] as String,
      name: json['name'] as String,
      latitude: (json['latitude'] as num).toDouble(),
      longitude: (json['longitude'] as num).toDouble(),
      distance: (json['distance'] as num).toDouble(),
      waitTime: json['waitTime'] as int,
      aiScore: (json['aiScore'] as num).toDouble(),
      reliability: (json['reliability'] as num).toDouble(),
      availableSlots: json['availableSlots'] as int,
      totalSlots: json['totalSlots'] as int,
      status: json['status'] as String,
      amenities: (json['amenities'] as List<dynamic>?)
          ?.map((e) => e as String)
          .toList() ?? [],
      imageUrl: json['imageUrl'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'latitude': latitude,
      'longitude': longitude,
      'distance': distance,
      'waitTime': waitTime,
      'aiScore': aiScore,
      'reliability': reliability,
      'availableSlots': availableSlots,
      'totalSlots': totalSlots,
      'status': status,
      'amenities': amenities,
      'imageUrl': imageUrl,
    };
  }
}

class QueueModel {
  final String id;
  final String stationId;
  final int position;
  final int estimatedWaitTime;
  final String status;
  final String? qrCode;
  final DateTime? joinedAt;
  final DateTime? estimatedCompletionTime;

  QueueModel({
    required this.id,
    required this.stationId,
    required this.position,
    required this.estimatedWaitTime,
    required this.status,
    this.qrCode,
    this.joinedAt,
    this.estimatedCompletionTime,
  });

  factory QueueModel.fromJson(Map<String, dynamic> json) {
    return QueueModel(
      id: json['id'] as String,
      stationId: json['stationId'] as String,
      position: json['position'] as int,
      estimatedWaitTime: json['estimatedWaitTime'] as int,
      status: json['status'] as String,
      qrCode: json['qrCode'] as String?,
      joinedAt: json['joinedAt'] != null
          ? DateTime.parse(json['joinedAt'] as String)
          : null,
      estimatedCompletionTime: json['estimatedCompletionTime'] != null
          ? DateTime.parse(json['estimatedCompletionTime'] as String)
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'stationId': stationId,
      'position': position,
      'estimatedWaitTime': estimatedWaitTime,
      'status': status,
      'qrCode': qrCode,
      'joinedAt': joinedAt?.toIso8601String(),
      'estimatedCompletionTime': estimatedCompletionTime?.toIso8601String(),
    };
  }
}
