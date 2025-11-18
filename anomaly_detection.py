import numpy as np

def detect_voltage_sag(voltage_data, threshold=7.2):
    """
    Detect voltage drops below safe threshold
    Returns: count of sag events and minimum voltage
    """
    if not voltage_data:
        return 0, None
    
    sag_events = [v for v in voltage_data if v < threshold]
    return len(sag_events), min(sag_events) if sag_events else None

def detect_vibration_spikes(vibration_data, z_threshold=3):
    """
    Detect vibration spikes using z-score
    z = (vibration - mean) / std
    if z > 3 → spike
    Returns: count of spikes and maximum spike value
    """
    if not vibration_data or len(vibration_data) < 2:
        return 0, None
    
    mean = np.mean(vibration_data)
    std = np.std(vibration_data)
    
    if std == 0:
        return 0, None
    
    spikes = [vib for vib in vibration_data if (vib - mean) / std > z_threshold]
    return len(spikes), max(spikes) if spikes else None

def detect_motor_imbalance(motor_outputs, imbalance_threshold=15):
    """
    Detect if one motor output is consistently higher/lower than others
    Returns: imbalance status and details
    """
    # Calculate average output for each motor
    motor_avgs = [(i+1, np.mean(motor)) for i, motor in enumerate(motor_outputs) if motor]
    
    if len(motor_avgs) < 2:
        return False, "Insufficient motor data"
    
    overall_avg = np.mean([avg for _, avg in motor_avgs])
    
    # Check for imbalance
    imbalances = []
    for motor_num, motor_avg in motor_avgs:
        deviation_pct = ((motor_avg - overall_avg) / overall_avg) * 100
        if abs(deviation_pct) > imbalance_threshold:
            imbalances.append(f"Motor {motor_num} is {deviation_pct:+.1f}% from average")
    
    return (True, "; ".join(imbalances)) if imbalances else (False, "All motors balanced")

def analyze_flight(data):
    """
    Run all anomaly detection algorithms on flight data
    Returns: dictionary with all detected anomalies
    """
    # Voltage sag detection
    sag_count, min_voltage = detect_voltage_sag(data['battery_voltage'])
    
    # Vibration spike detection
    vib_count, max_vibration = detect_vibration_spikes(data['vibration'])
    
    # Motor imbalance detection
    imbalanced, imbalance_msg = detect_motor_imbalance(data['motor_outputs'])
    
    return {
        'voltage_sag': {
            'detected': sag_count > 0,
            'count': sag_count,
            'min_voltage': min_voltage
        },
        'vibration_spikes': {
            'detected': vib_count > 0,
            'count': vib_count,
            'max_vibration': max_vibration
        },
        'motor_imbalance': {
            'detected': imbalanced,
            'message': imbalance_msg
        }
    }
