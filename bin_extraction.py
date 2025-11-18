from pymavlink import mavutil
import numpy as np

def read_bin_file(filepath):
    """Parse ArduPilot .BIN file using pymavlink"""
    mlog = mavutil.mavlink_connection(filepath)
    
    data = {
        'battery_voltage': [],
        'battery_current': [],
        'vibration': [],
        'gps_hdop': [],
        'motor_outputs': [[], [], [], []],
        'altitude': [],
        'timestamps': []
    }
    
    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break
        
        msg_type = msg.get_type()
        
        # Battery data
        if msg_type == 'BAT':
            data['battery_voltage'].append(msg.Volt)
            data['battery_current'].append(msg.Curr)
            data['timestamps'].append(msg.TimeUS / 1e6)
        
        # Vibration data
        elif msg_type == 'VIBE':
            vib_magnitude = np.sqrt(msg.VibeX**2 + msg.VibeY**2 + msg.VibeZ**2)
            data['vibration'].append(vib_magnitude)
        
        # GPS data
        elif msg_type == 'GPS':
            data['gps_hdop'].append(msg.HDop / 100.0)
        
        # Motor outputs
        elif msg_type == 'RCOU':
            for i in range(4):
                channel = getattr(msg, f'C{i+1}', None)
                if channel:
                    data['motor_outputs'][i].append(channel)
        
        # Altitude
        elif msg_type == 'BARO':
            data['altitude'].append(msg.Alt)
    
    return data