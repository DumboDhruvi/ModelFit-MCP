// Example: Consuming ModelFit Local Gateway in Flutter/Dart
//
// This allows your Flutter app to make plant disease predictions
// and swap models without ever modifying your UI or ViewModel code!

import 'dart:convert';
import 'package:http/http.dart' as http;

class PlantPrediction {
  final String label;
  final double score;

  PlantPrediction({required this.label, required this.score});

  factory PlantPrediction.fromJson(Map<String, dynamic> json) {
    return PlantPrediction(
      label: json['label'] as String,
      score: (json['score'] as num).toDouble(),
    );
  }
}

class ModelFitService {
  final String baseUrl;

  ModelFitService({this.baseUrl = 'http://127.0.0.1:7860'});

  /// Predict on an image path or base64 data
  Future<List<PlantPrediction>> identifyPlant(String imageInput) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'input': imageInput}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final list = data['predictions'] as List<dynamic>;
      return list.map((item) => PlantPrediction.fromJson(item as Map<String, dynamic>)).toList();
    } else {
      throw Exception('Prediction failed: ${response.body}');
    }
  }

  /// Hot-swap model (e.g. when user toggles 'High Accuracy' mode in settings)
  Future<void> upgradeToHighAccuracyModel() async {
    final response = await http.post(
      Uri.parse('$baseUrl/swap'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'model_id': 'google/vit-base-patch16-224',
        'task': 'image-classification'
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Hot-swap failed: ${response.body}');
    }
  }
}
