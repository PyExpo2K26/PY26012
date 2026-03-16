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

    try:
        # Initial pass for summary stats
        # Enabling IP defragmentation via tshark preferences
        # Note: 'ip.defragment' is a common tshark preference
        custom_parameters = ['-o', 'ip.defragment:TRUE']
        cap = pyshark.FileCapture(
            file_path, 
            tshark_path=TSHARK_PATH, 
            keep_packets=False,
            custom_parameters=custom_parameters
        )
        
        ips = Counter()
        protocols = Counter()
        domains = set()
        
        first_packet_time = None
        last_packet_time = None
        
        packet_count = 0
        
        # Detection flags
        cleartext_protocols = []
        suspicious_connections = 0
        scan_activity = 0
        c2_activity = 0
        fragmentation_detected = 0
        
        # New: Tracking for timeline and logs
        packet_log = []
        timeline_bins = Counter() # (int(timestamp)) -> count
        bytes_bins = Counter() # (int(timestamp)) -> total_bytes
        
        # Security tracking
        syn_packets = Counter() # src -> count
        icmp_packets = Counter() # src -> count

        for pkt in cap:
            packet_count += 1
            
            # Time tracking
            try:
                pkt_time = float(pkt.sniff_timestamp)
                if first_packet_time is None: first_packet_time = pkt_time
                last_packet_time = pkt_time
            except: pass

            # Protocol Tracking
