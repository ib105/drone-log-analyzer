import streamlit as st
import json
import numpy as np
from openai import OpenAI
import plotly.graph_objects as go
from datetime import datetime
from bin_extraction import read_bin_file
from anomaly_detection import analyze_flight

client = OpenAI(api_key="YOUR_API_KEY")

def get_openai_insights(data, anomaly_results):
    """Get AI insights from OpenAI GPT-5"""
    summary = {
        "flight_duration_sec": int(data['timestamps'][-1] - data['timestamps'][0]) if data['timestamps'] else 0,
        "battery_stats": {
            "min_voltage": round(min(data['battery_voltage']), 2) if data['battery_voltage'] else None,
            "max_voltage": round(max(data['battery_voltage']), 2) if data['battery_voltage'] else None,
            "avg_voltage": round(np.mean(data['battery_voltage']), 2) if data['battery_voltage'] else None,
            "avg_current": round(np.mean(data['battery_current']), 2) if data['battery_current'] else None,
        },
        "vibration_stats": {
            "max_vibration": round(max(data['vibration']), 2) if data['vibration'] else None,
            "avg_vibration": round(np.mean(data['vibration']), 2) if data['vibration'] else None,
        },
        "gps_stats": {
            "avg_hdop": round(np.mean(data['gps_hdop']), 2) if data['gps_hdop'] else None,
            "max_hdop": round(max(data['gps_hdop']), 2) if data['gps_hdop'] else None,
        },
        "motor_stats": {
            "motors_analyzed": len([m for m in data['motor_outputs'] if m]),
        },
        "altitude_stats": {
            "max_altitude": round(max(data['altitude']), 2) if data['altitude'] else None,
        },
        "anomalies": {
            "voltage_sag_events": anomaly_results['voltage_sag']['count'],
            "min_voltage_during_sag": anomaly_results['voltage_sag']['min_voltage'],
            "vibration_spikes": anomaly_results['vibration_spikes']['count'],
            "max_vibration_spike": anomaly_results['vibration_spikes']['max_vibration'],
            "motor_imbalance": anomaly_results['motor_imbalance']['message'],
        }
    }
    
    system_prompt = """You are an expert in drone flight analysis and telemetry interpretation. 
Your role is to analyze flight data and provide definitive technical assessments without asking follow-up questions.
Provide clear, actionable insights based solely on the data provided."""
    
    user_prompt = f"""Analyze the following drone flight telemetry data and provide a comprehensive technical assessment.

FLIGHT DATA:
{json.dumps(summary, indent=2)}

Provide your analysis in the following structured format:

1. OVERALL FLIGHT ASSESSMENT
   - Provide a concise summary of the flight's overall health and performance
   - Rate the flight quality (Excellent/Good/Fair/Poor/Critical)

2. CRITICAL ISSUES IDENTIFIED
   - List any safety-critical problems detected in the data
   - Explain the technical implications of each issue
   - If no critical issues exist, state "No critical issues detected"

3. TECHNICAL OBSERVATIONS
   - Battery performance analysis (voltage behavior, sag events, capacity concerns)
   - Vibration analysis (mechanical health, potential causes)
   - Motor performance (balance, output consistency, thermal considerations)
   - Flight stability indicators

4. MAINTENANCE RECOMMENDATIONS
   - Provide specific, actionable maintenance tasks based on the data
   - Prioritize recommendations by urgency (Immediate/Short-term/Routine)
   - If no maintenance is needed, state "No immediate maintenance required"

5. RISK ASSESSMENT
   - Overall risk level: LOW / MODERATE / HIGH / CRITICAL
   - Justification for the risk rating
   - Flight worthiness status (Safe to fly / Requires attention / Grounded)

Provide definitive answers based on the available data. Do not ask for additional information or pose questions."""
    
    response = client.chat.completions.create(
        model="gpt-5-search-api-2025-10-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

def create_graph(x, y, title, ylabel, color):
    """Create a simple plotly graph"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=2)))
    fig.update_layout(title=title, xaxis_title='Time (s)', yaxis_title=ylabel, template='plotly_dark')
    return fig

def generate_report(data, anomaly_results, insights, filename):
    """Generate text report"""
    lines = [
        "DRONE LOG ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"File: {filename}\n",
        "STATISTICS:",
        f"Duration: {data['timestamps'][-1] - data['timestamps'][0]:.1f}s" if data['timestamps'] else "",
        f"Voltage: {np.mean(data['battery_voltage']):.2f}V (avg)" if data['battery_voltage'] else "",
        f"Vibration: {np.mean(data['vibration']):.2f} m/s² (avg)" if data['vibration'] else "",
        "\nANOMALIES:",
        f"Voltage Sags: {anomaly_results['voltage_sag']['count']}" if anomaly_results['voltage_sag']['detected'] else "None",
        f"Vibration Spikes: {anomaly_results['vibration_spikes']['count']}" if anomaly_results['vibration_spikes']['detected'] else "",
        f"Motor Imbalance: {anomaly_results['motor_imbalance']['message']}" if anomaly_results['motor_imbalance']['detected'] else "",
        "\nAI INSIGHTS:",
        insights
    ]
    return "\n".join([l for l in lines if l])

st.set_page_config(page_title="Drone Analyzer", layout="wide")
st.title("Drone Log Analyzer")

uploaded_file = st.file_uploader("Upload .BIN file", type=['bin', 'BIN'])

if uploaded_file and st.button("Analyze", type="primary"):
    with open("temp.bin", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing..."):
        data = read_bin_file("temp.bin")
        anomaly_results = analyze_flight(data)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        if data['timestamps']:
            col1.metric("Duration", f"{data['timestamps'][-1] - data['timestamps'][0]:.1f}s")
        if data['battery_voltage']:
            col2.metric("Avg Voltage", f"{np.mean(data['battery_voltage']):.2f}V")
        if data['vibration']:
            col3.metric("Avg Vibration", f"{np.mean(data['vibration']):.2f} m/s²")
        
        # Anomalies
        st.subheader("Anomalies Detected")
        if anomaly_results['voltage_sag']['detected']:
            st.error(f"Voltage Sag: {anomaly_results['voltage_sag']['count']} events")
        if anomaly_results['vibration_spikes']['detected']:
            st.warning(f"Vibration Spikes: {anomaly_results['vibration_spikes']['count']} events")
        if anomaly_results['motor_imbalance']['detected']:
            st.warning(f"Motor Imbalance: {anomaly_results['motor_imbalance']['message']}")
        if not any([anomaly_results['voltage_sag']['detected'], anomaly_results['vibration_spikes']['detected'], anomaly_results['motor_imbalance']['detected']]):
            st.success("No anomalies detected")
        
        # Graphs
        st.subheader("Flight Data Visualization")
        if data['battery_voltage'] and data['timestamps']:
            st.plotly_chart(create_graph(data['timestamps'], data['battery_voltage'], 'Battery Voltage', 'Voltage (V)', '#00cc96'), width='stretch')
        
        if data['vibration']:
            times = np.linspace(data['timestamps'][0], data['timestamps'][-1], len(data['vibration'])) if data['timestamps'] else list(range(len(data['vibration'])))
            st.plotly_chart(create_graph(times, data['vibration'], 'Vibration', 'Vibration (m/s²)', '#ff6692'), width='stretch')
        
        # Motor outputs
        if any(data['motor_outputs']):
            fig = go.Figure()
            colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa']
            for i, motor in enumerate(data['motor_outputs']):
                if motor:
                    times = np.linspace(data['timestamps'][0], data['timestamps'][-1], len(motor)) if data['timestamps'] else list(range(len(motor)))
                    fig.add_trace(go.Scatter(x=times, y=motor, mode='lines', name=f'Motor {i+1}', line=dict(color=colors[i])))
            fig.update_layout(title='Motor Outputs', xaxis_title='Time (s)', yaxis_title='PWM', template='plotly_dark')
            st.plotly_chart(fig, width='stretch')
        
        # AI Insights
        st.subheader("GPT5 Insights")
        try:
            insights = get_openai_insights(data, anomaly_results)
            st.markdown(insights)
        except Exception as e:
            st.error(f"Error getting AI insights: {str(e)}")
            insights = "AI analysis unavailable"
        
        # Download
        report = generate_report(data, anomaly_results, insights, uploaded_file.name)
        st.download_button("Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", type="primary")
