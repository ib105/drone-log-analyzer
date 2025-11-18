import numpy as np
from bin_extraction import read_bin_file

def detect_voltage_sag(voltage_data, threshold=7.2):
    """
    Detect voltage drops below safe threshold
    Returns: count of sag events and list of voltage values below threshold
    """
    if not voltage_data:
        return 0, []
    
    sag_events = [v for v in voltage_data if v < threshold]
    return len(sag_events), sag_events

def detect_vibration_spikes(vibration_data, z_threshold=3):
    """
    Detect vibration spikes using z-score
    z = (vibration - mean) / std
    if z > 3 → spike
    Returns: count of spikes and list of spike values
    """
    if not vibration_data or len(vibration_data) < 2:
        return 0, []
    
    mean = np.mean(vibration_data)
    std = np.std(vibration_data)
    
    if std == 0:
        return 0, []
    
    spikes = []
    for vib in vibration_data:
        z_score = (vib - mean) / std
        if z_score > z_threshold:
            spikes.append(vib)
    
    return len(spikes), spikes

def detect_motor_imbalance(motor_outputs, imbalance_threshold=15):
    """
    Detect if one motor output is consistently higher/lower than others
    Returns: imbalance status and details
    """
    if not any(motor_outputs):
        return False, "No motor data available"
    
    # Calculate average output for each motor
    motor_avgs = []
    for i, motor in enumerate(motor_outputs):
        if motor:
            motor_avgs.append((i+1, np.mean(motor)))
    
    if len(motor_avgs) < 2:
        return False, "Insufficient motor data"
    
    # Calculate overall average
    overall_avg = np.mean([avg for _, avg in motor_avgs])
    
    # Check for imbalance
    imbalances = []
    for motor_num, motor_avg in motor_avgs:
        deviation_pct = ((motor_avg - overall_avg) / overall_avg) * 100
        if abs(deviation_pct) > imbalance_threshold:
            imbalances.append(f"Motor {motor_num} is {deviation_pct:+.1f}% from average")
    
    if imbalances:
        return True, "; ".join(imbalances)
    else:
        return False, "All motors balanced"

def analyze_flight(data):
    """
    Run all anomaly detection algorithms on flight data
    Returns: dictionary with all detected anomalies
    """
    results = {}
    
    # Voltage sag detection
    sag_count, sag_values = detect_voltage_sag(data['battery_voltage'])
    results['voltage_sag'] = {
        'detected': sag_count > 0,
        'count': sag_count,
        'min_voltage': min(sag_values) if sag_values else None
    }
    
    # Vibration spike detection
    vib_count, vib_spikes = detect_vibration_spikes(data['vibration'])
    results['vibration_spikes'] = {
        'detected': vib_count > 0,
        'count': vib_count,
        'max_vibration': max(vib_spikes) if vib_spikes else None
    }
    
    # Motor imbalance detection
    imbalanced, imbalance_msg = detect_motor_imbalance(data['motor_outputs'])
    results['motor_imbalance'] = {
        'detected': imbalanced,
        'message': imbalance_msg
    }
    
    return results