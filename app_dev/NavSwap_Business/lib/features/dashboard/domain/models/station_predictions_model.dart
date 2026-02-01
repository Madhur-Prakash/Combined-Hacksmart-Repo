class StationPredictionsModel {
  final LoadForecast loadForecast;
  final FaultPrediction faultPrediction;

  StationPredictionsModel({
    required this.loadForecast,
    required this.faultPrediction,
  });

  factory StationPredictionsModel.fromJson(Map<String, dynamic> json) {
    return StationPredictionsModel(
      loadForecast: LoadForecast.fromJson(json['loadForecast']),
      faultPrediction: FaultPrediction.fromJson(json['faultPrediction']),
    );
  }
}

class LoadForecast {
  final String stationId;
  final double predictedLoad;
  final double confidence;
  final String peakTimeStart;
  final String peakTimeEnd;
  final int timestamp;

  LoadForecast({
    required this.stationId,
    required this.predictedLoad,
    required this.confidence,
    required this.peakTimeStart,
    required this.peakTimeEnd,
    required this.timestamp,
  });

  factory LoadForecast.fromJson(Map<String, dynamic> json) {
    return LoadForecast(
      stationId: json['stationId'],
      predictedLoad: json['predictedLoad'],
      confidence: json['confidence'],
      peakTimeStart: json['peakTimeStart'],
      peakTimeEnd: json['peakTimeEnd'],
      timestamp: json['timestamp'],
    );
  }
}

class FaultPrediction {
  final String stationId;
  final double faultProbability;
  final String? predictedFaultType;
  final String riskLevel;
  final double confidence;
  final int timestamp;

  FaultPrediction({
    required this.stationId,
    required this.faultProbability,
    this.predictedFaultType,
    required this.riskLevel,
    required this.confidence,
    required this.timestamp,
  });

  factory FaultPrediction.fromJson(Map<String, dynamic> json) {
    return FaultPrediction(
      stationId: json['stationId'],
      faultProbability: json['faultProbability'],
      predictedFaultType: json['predictedFaultType'],
      riskLevel: json['riskLevel'],
      confidence: json['confidence'],
      timestamp: json['timestamp'],
    );
  }
}
