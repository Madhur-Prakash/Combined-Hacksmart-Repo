# HackSmart Hackathon - Combined Repository

This repository contains the complete solution for the HackSmart hackathon, featuring a comprehensive EV battery swapping ecosystem with backend services, mobile applications, and frontend interfaces.

## 🏗️ Repository Structure

### 📱 App Development (`app_dev/`)
- **NavSwap**: Main Flutter mobile application for EV battery swapping
- **NavSwap_AI_Service**: AI service backend for the mobile app
- **NavSwap_Business**: Business-focused Flutter application

### 🔧 Backend Services (`hacksmart/`)
- **Core API**: Main backend services with microservices architecture
- **Smart EV Swap Map**: React-based frontend for station management
- **Docker Services**: Containerized microservices for scalability
- **ML Models**: Pre-trained models for demand prediction and optimization

### 🤖 AI Services (`Hacksmart_Nexora_Navswap_AI/`)
- Advanced AI models for battery management
- Fault detection and prediction systems
- Real-time analytics and recommendations

### 🎥 Video API (`video-api/`)
- Video processing and streaming services
- Media handling for the application ecosystem

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js (v16+)
- Python (3.8+)
- Flutter SDK
- PostgreSQL
- Redis

### Backend Setup
```bash
cd hacksmart
docker-compose up -d
```

### Mobile App Setup
```bash
cd app_dev/NavSwap
flutter pub get
flutter run
```

### AI Services Setup
```bash
cd Hacksmart_Nexora_Navswap_AI
pip install -r requirements.txt
python app/main.py
```

## 🏆 Hackathon Features

- **Smart Battery Swapping**: Automated EV battery exchange system
- **AI-Powered Optimization**: Machine learning for demand prediction
- **Real-time Monitoring**: Live tracking of battery stations and inventory
- **Mobile Applications**: User and business-focused mobile apps
- **Scalable Architecture**: Microservices-based backend infrastructure

## 📚 Documentation

Each component has detailed documentation in its respective directory:
- [Backend API Documentation](hacksmart/README.md)
- [Mobile App Guide](app_dev/NavSwap/README.md)
- [AI Services Documentation](Hacksmart_Nexora_Navswap_AI/README.md)

## 🛠️ Technology Stack

- **Backend**: Node.js, TypeScript, PostgreSQL, Redis, Kafka
- **Mobile**: Flutter, Dart
- **AI/ML**: Python, FastAPI, scikit-learn, XGBoost
- **Frontend**: React, TypeScript
- **Infrastructure**: Docker, Docker Compose

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](hacksmart/LICENSE) file for details.

---

**Team**: Nexora NavSwap  
**Event**: HackSmart Hackathon  
**Year**: 2024