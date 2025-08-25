import pandas as pd
import numpy as np
from src.matrix_tracking.system import MatrixTrackingSystem


def simulate_vehicle_movement(
    matrix_tracking,
    vehicle_id,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    departure_time,
    speed_factor=1.0,
    deviation=0.0,
):
    """Simula o movimento de um veículo ao longo da rota"""
    # Planejar a rota
    plan_result = matrix_tracking.plan_route(
        vehicle_id, start_lat, start_lng, end_lat, end_lng, departure_time
    )

    print(f"\nVeículo {vehicle_id} planejado:")
    print(f"  Rota: {start_lat:.4f},{start_lng:.4f} → {end_lat:.4f},{end_lng:.4f}")
    print(f"  Duração estimada: {plan_result['expected_duration']/60:.1f} minutos")
    print(f"  Chegada prevista: {plan_result['expected_arrival'].strftime('%H:%M:%S')}")

    # Obter a rota
    route = plan_result["planned_route"]
    waypoints = route["waypoints"]

    # Simular movimento com possível desvio
    if deviation > 0:
        # Adicionar um desvio no meio da rota
        mid_point = len(waypoints) // 2
        for i in range(mid_point - 2, mid_point + 3):
            if i > 0 and i < len(waypoints) - 1:
                # Desvio perpendicular à rota
                dx = waypoints[i + 1][1] - waypoints[i - 1][1]
                dy = waypoints[i + 1][0] - waypoints[i - 1][0]
                # Perpendicular: (-dy, dx)
                norm = np.sqrt(dx * dx + dy * dy)
                if norm > 0:
                    waypoints[i] = (
                        waypoints[i][0] + deviation * (-dy / norm),
                        waypoints[i][1] + deviation * (dx / norm),
                    )

    # Criar timestamps intermediários baseados no speed_factor
    departure_dt = pd.to_datetime(departure_time)
    total_duration = plan_result["expected_duration"] / speed_factor
    timestamps = []
    for i in range(len(waypoints)):
        frac = i / (len(waypoints) - 1)
        ts = departure_dt + pd.Timedelta(seconds=total_duration * frac)
        timestamps.append(ts)

    # Simular o movimento enviando atualizações
    updates = []
    for i in range(len(waypoints)):
        update = matrix_tracking.update_vehicle_position(
            vehicle_id, waypoints[i][0], waypoints[i][1], timestamps[i]
        )
        updates.append(update)

        # Se for o último ponto, verificar se completou
        if i == len(waypoints) - 1:
            print(f"Veículo {vehicle_id} chegou ao destino:")
            print(f"  Status: {update['status']}")
            if "actual_duration" in update:
                print(f"  Duração real: {update['actual_duration']/60:.1f} minutos")
                print(f"  Desvio: {update['deviation']:.1f}%")
                if update["is_anomaly"]:
                    print("  ALERTA: Anomalia de tempo detectada!")
        # Se for um ponto intermediário com alerta, mostrar
        elif "alerts" in update and update["alerts"]:
            print(f"  Alerta em {timestamps[i].strftime('%H:%M:%S')}:")
            print(f"    {update['alerts'][-1]['details']}")

    return updates


def run_demo():
    """Executa uma demonstração do sistema Matrix Tracking"""
    print(
        "\n==== Sistema Matrix Tracking: Monitoramento em Tempo Real de Veículos de Carga ===="
    )

    # Inicializar o sistema
    matrix_tracking = MatrixTrackingSystem()

    # Carregar alguns dados de exemplo (podem vir do seu conjunto de dados)
    sample_data = [
        # Exemplo 1: Viagem normal
        {
            "id": "TRUCK-001",
            "start_lat": 37.7749,
            "start_lng": -122.4194,
            "end_lat": 37.3382,
            "end_lng": -121.8863,
            "departure": "2023-05-15 08:30:00",
            "speed_factor": 1.0,
            "deviation": 0.0,
        },
        # Exemplo 2: Viagem com atraso
        {
            "id": "TRUCK-002",
            "start_lat": 37.7749,
            "start_lng": -122.4194,
            "end_lat": 37.8716,
            "end_lng": -122.2727,
            "departure": "2023-05-15 17:30:00",  # Hora de pico
            "speed_factor": 0.6,  # 40% mais lento que o esperado
            "deviation": 0.0,
        },
        # Exemplo 3: Viagem com desvio de rota
        {
            "id": "TRUCK-003",
            "start_lat": 37.7749,
            "start_lng": -122.4194,
            "end_lat": 37.4275,
            "end_lng": -122.1697,
            "departure": "2023-05-15 12:30:00",
            "speed_factor": 0.9,  # Ligeiramente mais lento
            "deviation": 0.3,  # Desvio moderado da rota
        },
    ]

    # Simular o movimento de cada veículo
    for vehicle_data in sample_data:
        simulate_vehicle_movement(
            matrix_tracking,
            vehicle_data["id"],
            vehicle_data["start_lat"],
            vehicle_data["start_lng"],
            vehicle_data["end_lat"],
            vehicle_data["end_lng"],
            vehicle_data["departure"],
            vehicle_data["speed_factor"],
            vehicle_data["deviation"],
        )

    # Mostrar alertas gerados
    print("\n==== Alertas Gerados pelo Sistema ====")
    for i, alert in enumerate(matrix_tracking.alerts):
        print(f"Alerta {i+1}:")
        print(f"  Veículo: {alert['vehicle_id']}")
        print(f"  Tipo: {alert['type']}")
        print(f"  Tempo: {alert['timestamp'].strftime('%H:%M:%S')}")
        print(f"  Detalhes: {alert['details']}")

    # Mostrar estatísticas do banco de dados de trajetórias
    print("\n==== Estatísticas do Banco de Dados de Trajetórias ====")
    stats = matrix_tracking.trajectory_db.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    run_demo()
