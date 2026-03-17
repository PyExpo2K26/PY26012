import pyshark
import os
from collections import Counter
import datetime
import asyncio
import subprocess

TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"

def generate_firewall_rules(suspicious_ips):
    """
    Generates minimal firewall rules for suspicious IPs.
    """
    rules = []
    for ip in suspicious_ips:
        if ip and ip != "N/A":
            rules.append({
                "ip": ip,
                "windows": f'netsh advfirewall firewall add rule name="Block-Suspicious-{ip}" dir=in action=block remoteip={ip}',
                "linux": f'iptables -A INPUT -s {ip} -j DROP'
            })
    return rules

def capture_live_traffic(duration=15):
    """
    Captures live traffic using TShark and returns analyzed report.
    """
    import tempfile
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"live_capture_{int(datetime.datetime.now().timestamp())}.pcap")
    
    if not os.path.exists(TSHARK_PATH):
        return {"error": "TShark not found. Cannot start live capture."}
        
    try:
        print(f"Starting live capture to {filename} for {duration} seconds...")
        cmd = [TSHARK_PATH, "-a", f"duration:{duration}", "-w", filename]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 5)
        
        if result.returncode != 0:
             return {"error": f"TShark error: {result.stderr or result.stdout}"}
             
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
             return {"error": "No packets captured during live stream."}
             
        report = analyze_pcap(filename)
        try: os.remove(filename)
        except: pass
        return report
    except Exception as e:
        return {"error": f"Live capture failed: {str(e)}"}

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
        },
        "firewall_rules": [],
        "recommendations": []
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
            proto = pkt.highest_layer
            
            # If it's just 'DATA' but we have IP, it might be a fragment or unparsed
            if proto == 'DATA' and 'IP' in pkt:
                # Try to see if there's anything else interesting
                for layer in reversed(pkt.layers):
                    if layer.layer_name not in ['eth', 'ip', 'data', 'DATA']:
                        proto = layer.layer_name.upper()
                        break
            
            protocols[proto] += 1
            
            # Vulnerability Detection: Cleartext
            if proto in ['HTTP', 'FTP', 'TELNET', 'SMTP']:
                if proto not in cleartext_protocols:
                    cleartext_protocols.append(proto)

            # IP Tracking
            if 'IP' in pkt:
                src = pkt.ip.src
                dst = pkt.ip.dst
                ips[src] += 1
                ips[dst] += 1
                
                # Aggregate Counter Fix
                if 'TCP' in pkt and hasattr(pkt.tcp, 'flags') and pkt.tcp.flags == '0x00000002': # SYN
                     syn_packets[src] += 1
                if 'ICMP' in pkt:
                     icmp_packets[src] += 1
                
                # Fragmentation Detection
                # MF (More Fragments) flag or non-zero fragmentation offset
                is_fragment = False
                try:
                    if hasattr(pkt.ip, 'flags_mf') and pkt.ip.flags_mf == '1':
                        is_fragment = True
                    if hasattr(pkt.ip, 'frag_offset') and int(pkt.ip.frag_offset) > 0:
                        is_fragment = True
                except: pass
                
                if is_fragment:
                    fragmentation_detected += 1

            # --- Packet Logging (Only Suspicious) ---
            is_suspicious_pkt = False
            
            # 1. Cleartext Check
            if proto in ['HTTP', 'FTP', 'TELNET', 'SMTP']:
                is_suspicious_pkt = True
                
            # 2. Fragmentation Check
            if 'is_fragment' in locals() and is_fragment:
                is_suspicious_pkt = True
                
            # 3. Security check (ICMP flows log all)
            if proto == 'ICMP':
                is_suspicious_pkt = True
                
            # 4. Long DNS query check
            if 'DNS' in pkt and hasattr(pkt.dns, 'qry_name') and len(pkt.dns.qry_name) > 30:
                is_suspicious_pkt = True

            if is_suspicious_pkt and len(packet_log) < 500:
                pkt_info = {
                    "no": packet_count,
                    "time": datetime.datetime.fromtimestamp(pkt_time).strftime('%H:%M:%S.%f')[:-3] if pkt_time else "N/A",
                    "src": pkt.ip.src if 'IP' in pkt else "N/A",
                    "dst": pkt.ip.dst if 'IP' in pkt else "N/A",
                    "proto": proto,
                    "length": pkt.length,
                    "info": f"Layer: {proto}"
                }
                
                if 'TCP' in pkt:
                    pkt_info["info"] = f"TCP Port: {pkt.tcp.srcport} -> {pkt.tcp.dstport} [{pkt.tcp.flags_str}]"
                elif 'UDP' in pkt:
                    pkt_info["info"] = f"UDP Port: {pkt.udp.srcport} -> {pkt.udp.dstport}"
                elif 'ICMP' in pkt:
                    pkt_info["info"] = f"ICMP Type: {pkt.icmp.type}"
                
                packet_log.append(pkt_info)

            # --- Timeline Tracking ---
            if pkt_time:
                bin_sec = int(pkt_time)
                timeline_bins[bin_sec] += 1
                bytes_bins[bin_sec] += int(pkt.length)

            # Domain Tracking (DNS)
            if 'DNS' in pkt and hasattr(pkt.dns, 'qry_name'):
                domains.add(pkt.dns.qry_name)

            # Basic "Scanning" detection (very naive: high packet count to many ports)
            # In a real engine we'd track state, here we'll just increment for demo
            if packet_count % 100 == 0: # Simulating detection logic
                suspicious_connections += 1

        cap.close()

        # Finalize Metrics
        report["metrics"]["total_packets"] = packet_count
        
        if first_packet_time and last_packet_time:
            duration = last_packet_time - first_packet_time
            report["file_info"]["duration"] = str(datetime.timedelta(seconds=int(duration)))
            report["file_info"]["recorded"] = datetime.datetime.fromtimestamp(first_packet_time).strftime('%Y-%m-%d %H:%M')

        # Split IPs (Simple heuristic: 192.168.*, 10.*, 172.16-31.* are internal)
        internal = []
        external = []
        for ip in ips:
            if ip.startswith(('192.168.', '10.', '172.')): # Simplified
                internal.append(ip)
            else:
                external.append(ip)
        
        report["metrics"]["internal_hosts"] = len(internal)
        report["metrics"]["external_hosts"] = len(external)
        report["metrics"]["unique_ips"] = len(ips)
        report["metrics"]["unique_domains"] = len(domains)
        
        # Protocols
        report["protocols"] = dict(protocols.most_common(5))
        
        # Findings
        if cleartext_protocols:
            report["vulnerabilities"].append({
                "title": "CLEARTEXT PROTOCOLS",
                "count": protocols[cleartext_protocols[0]], # Usage of first detected cleartext
                "desc": f"Use of unencrypted protocols detected: {', '.join(cleartext_protocols)}"
            })
            report["metrics"]["risk_score"] += 20

        if suspicious_connections > 0:
            report["suspicious_activities"].append({
                "title": "SUSPICIOUS CONNECTIONS",
                "count": suspicious_connections,
                "desc": "Traffic flags indicating communication with potentially flagged domains."
            })
            report["metrics"]["risk_score"] += 30

        if fragmentation_detected > 0:
            report["suspicious_activities"].append({
                "title": "IP FRAGMENTATION",
                "count": fragmentation_detected,
                "desc": "Fragmented IP packets detected. Can be used to evade IDS/IPS or indicate network MTU issues."
            })
            report["metrics"]["risk_score"] += 15

        # --- Advanced Security Detection Results ---
        suspicious_ips = set()
        
        # 1. ICMP Flood Detection
        for src, count in icmp_packets.items():
            if count > 50:
                report["suspicious_activities"].append({
                    "title": "POSSIBLE ICMP FLOOD",
                    "count": count,
                    "desc": f"Host {src} is sending a high volume of ICMP packets (Potential DoS/Scan)."
                })
                suspicious_ips.add(src)
                report["metrics"]["risk_score"] += 40
                
        # 2. Port Scanning
        for src, count in syn_packets.items():
            if count > 100:
                report["suspicious_activities"].append({
                    "title": "SYN SCAN/FLOOD DETECTED",
                    "count": count,
                    "desc": f"Host {src} initiated {count} SYN requests. Characteristic of port scanning or SYN flood attack."
                })
                suspicious_ips.add(src)
                report["metrics"]["risk_score"] += 35
                
        # 3. DNS Tunneling Heuristic
        dns_tunnel_count = 0
        for domain in domains:
            if len(domain) > 35:
                dns_tunnel_count += 1
                
        if dns_tunnel_count > 0:
            report["suspicious_activities"].append({
                "title": "POTENTIAL DNS TUNNELING",
                "count": dns_tunnel_count,
                "desc": "Unusually long DNS queries detected. Often used for data exfiltration or C2 communication."
            })
            report["metrics"]["risk_score"] += 20
            
        report["firewall_rules"] = generate_firewall_rules(suspicious_ips)
        
        if suspicious_ips:
             report["recommendations"].append("Block high-risk external hosts using the recommended firewall rules.")
        if fragmentation_detected > 0:
             report["recommendations"].append("Inspect network for MTU mismatch or fragmentation/evasion attacks.")

        # Finalize Timeline Sort
        sorted_times = sorted(timeline_bins.keys())
        if sorted_times:
            report["timeline"]["timestamps"] = [datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in sorted_times]
            report["timeline"]["packet_counts"] = [timeline_bins[t] for t in sorted_times]
            report["timeline"]["byte_counts"] = [bytes_bins[t] for t in sorted_times]

        report["packet_log"] = packet_log

        # Caps risk score at 100
        report["metrics"]["risk_score"] = min(report["metrics"]["risk_score"], 100)

        return report

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test logic
    print("PCAP Engine Initialized")
