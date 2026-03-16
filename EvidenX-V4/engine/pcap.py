import pyshark
import os
from collections import Counter
import datetime
import asyncio

TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"

def analyze_pcap(file_path):
    """
    Analyzes a PCAP file and returns a structured report for the cybersecurity dashboard.
    """
    # Ensure there's an event loop in this thread (needed by pyshark)
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not os.path.exists(TSHARK_PATH):
        return {"error": "TShark not found at specified path. Please ensure Wireshark is installed."}

    report = {
        "file_info": {
            "name": os.path.basename(file_path),
            "size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "duration": "00:00:00",
            "recorded": "Unknown"
        },
        "metrics": {
            "total_packets": 0,
            "internal_hosts": 0,
            "external_hosts": 0,
            "unique_ips": 0,
            "unique_domains": 0,
            "risk_score": 0
        },
        "protocols": {},
        "suspicious_activities": [],
        "vulnerabilities": [],
        "endpoints": {
            "internal_ips": [],
            "external_ips": []
        },
        "packet_log": [],
        "timeline": {
            "timestamps": [],
            "packet_counts": [],
            "byte_counts": []
        }
    }
