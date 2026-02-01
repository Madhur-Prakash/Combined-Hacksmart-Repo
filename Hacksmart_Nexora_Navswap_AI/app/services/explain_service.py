import google.generativeai as genai
from app.utils.config import GEMINI_API_KEY
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ExplanationService:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
    
    async def explain_decision(self, action: str, queue: float, wait_time: float, 
                             fault_prob: float, reliability: float, energy: float) -> str:
        try:
            prompt = f"""
            Explain why the NavSwap AI system recommended action '{action}' for an EV battery swap station.
            
            Current Predictions:
            - Queue Length: {queue:.1f} vehicles
            - Wait Time: {wait_time:.1f} minutes
            - Fault Probability: {fault_prob:.2%}
            - Station Reliability: {reliability:.2%}
            - Energy Stability: {energy:.2%}
            
            Provide a clear, concise explanation (2-3 sentences) for why this action was chosen to optimize the EV battery swap system.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"AI decision based on current system metrics: Queue={queue:.1f}, Wait={wait_time:.1f}min, Fault Risk={fault_prob:.2%}. Action '{action}' optimizes station performance."
    
    async def explain_operations(self, operations_data: Dict[str, Any]) -> str:
        """Generate comprehensive explanation for all operational decisions"""
        try:
            # Extract key metrics
            station = operations_data.get('station_recommendation', {})
            traffic = operations_data.get('traffic_prediction', {})
            battery_plan = operations_data.get('battery_transport_plan', {})
            staff_plan = operations_data.get('staff_diversion_plan', {})
            inventory_plan = operations_data.get('inventory_order_plan', {})
            storage_plan = operations_data.get('partner_storage_plan', {})
            
            prompt = f"""
            You are NavSwap's AI Operations Assistant. Explain the comprehensive operational strategy based on multiple AI model predictions.
            
            STATION RECOMMENDATION:
            - Selected: {station.get('station_name', 'N/A')}
            - Queue Prediction: {station.get('predicted_queue', 0):.1f} vehicles
            - Wait Time: {station.get('predicted_wait', 0):.1f} minutes
            - System Action: {station.get('system_action', 'NORMAL')}
            
            TRAFFIC ANALYSIS:
            - Micro Traffic Score: {traffic.get('micro_traffic_score', 0):.2f}
            - Congestion Level: {traffic.get('congestion_level', 'MEDIUM')}
            - Future Traffic Trend: {traffic.get('traffic_trend', 'STABLE')}
            
            BATTERY LOGISTICS:
            - Rebalance Needed: {battery_plan.get('rebalance_needed', False)}
            - Batteries to Move: {battery_plan.get('batteries_to_move', 0)}
            - Priority: {battery_plan.get('priority', 'MEDIUM')}
            
            STAFF OPERATIONS:
            - Staff Diversion Needed: {staff_plan.get('diversion_needed', False)}
            - Staff Count: {staff_plan.get('staff_count', 0)}
            - Priority: {staff_plan.get('priority', 'MEDIUM')}
            
            INVENTORY MANAGEMENT:
            - Order Needed: {inventory_plan.get('order_needed', False)}
            - Battery Quantity: {inventory_plan.get('battery_quantity', 0)}
            - Urgency: {inventory_plan.get('urgency', 'MEDIUM')}
            
            PARTNER STORAGE:
            - Storage Needed: {storage_plan.get('storage_needed', False)}
            - Partner Stations: {len(storage_plan.get('partner_stations', []))}
            
            Provide a comprehensive, friendly explanation (4-5 sentences) that connects all these operational decisions and explains how they work together to optimize the EV battery swap network.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            logger.warning(f"Operations explanation failed: {e}")
            return self._generate_fallback_operations_explanation(operations_data)
    
    def _generate_fallback_operations_explanation(self, operations_data: Dict[str, Any]) -> str:
        """Fallback explanation when Gemini API fails"""
        station = operations_data.get('station_recommendation', {})
        traffic = operations_data.get('traffic_prediction', {})
        
        station_name = station.get('station_name', 'the selected station')
        congestion = traffic.get('congestion_level', 'MEDIUM')
        
        return f"NavSwap AI recommends {station_name} based on comprehensive analysis of traffic patterns ({congestion} congestion), battery logistics, staff availability, and inventory levels. The system has coordinated all operational aspects to ensure optimal service delivery and resource utilization across the network."