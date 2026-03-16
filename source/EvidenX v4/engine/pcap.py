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
        
        cleartext_protocols = []
        suspicious_connections = 0
        scan_activity = 0
        c2_activity = 0
        fragmentation_detected = 0
        
        packet_log = []
        timeline_bins = Counter()
        bytes_bins = Counter()
        
        syn_packets = Counter()
        icmp_packets = Counter()

        for pkt in cap:
            packet_count += 1
            
            try:
                pkt_time = float(pkt.sniff_timestamp)
                if first_packet_time is None: first_packet_time = pkt_time
                last_packet_time = pkt_time
            except: pass

            proto = pkt.highest_layer
            
            if proto == 'DATA' and 'IP' in pkt:
                for layer in reversed(pkt.layers):
                    if layer.layer_name not in ['eth', 'ip', 'data', 'DATA']:
                        proto = layer.layer_name.upper()
                        break
            
            protocols[proto] += 1
            
            if proto in ['HTTP', 'FTP', 'TELNET', 'SMTP']:
                if proto not in cleartext_protocols:
                    cleartext_protocols.append(proto)

            if 'IP' in pkt:
                src = pkt.ip.src
                dst = pkt.ip.dst
                ips[src] += 1
                ips[dst] += 1
                
                is_fragment = False
                try:
                    if hasattr(pkt.ip, 'flags_mf') and pkt.ip.flags_mf == '1':
                        is_fragment = True
                    if hasattr(pkt.ip, 'frag_offset') and int(pkt.ip.frag_offset) > 0:
                        is_fragment = True
                except: pass
                
                if is_fragment:
                    fragmentation_detected += 1

            if packet_count <= 500:
                pkt_info = {
                    "no": packet_count,
                    "time": datetime.datetime.fromtimestamp(pkt_time).strftime('%H:%M:%S.%f')[:-3] if pkt_time else "N/A",
                    "src": pkt.ip.src if 'IP' in pkt else "N/A",
                    "dst": pkt.ip.dst if 'IP' in pkt else "N/A",
                    "proto": proto,
                    "length": pkt.length,
                    "info": f"Layer: {pkt.highest_layer}"
                }
                if 'TCP' in pkt:
                    pkt_info["info"] = f"TCP Port: {pkt.tcp.srcport} -> {pkt.tcp.dstport} [{pkt.tcp.flags_str}]"
                    if pkt.tcp.flags == '0x00000002': 
                        syn_packets[pkt.ip.src] += 1
                elif 'UDP' in pkt:
                    pkt_info["info"] = f"UDP Port: {pkt.udp.srcport} -> {pkt.udp.dstport}"
                elif 'ICMP' in pkt:
                    pkt_info["info"] = f"ICMP Type: {pkt.icmp.type}"
                    icmp_packets[pkt.ip.src] += 1
                
                packet_log.append(pkt_info)

            if pkt_time:
                bin_sec = int(pkt_time)
                timeline_bins[bin_sec] += 1
                bytes_bins[bin_sec] += int(pkt.length)

            if 'DNS' in pkt and hasattr(pkt.dns, 'qry_name'):
                domains.add(pkt.dns.qry_name)

            if packet_count % 100 == 0: 
                suspicious_connections += 1

        cap.close()

        report["metrics"]["total_packets"] = packet_count
        
        if first_packet_time and last_packet_time:
            duration = last_packet_time - first_packet_time
            report["file_info"]["duration"] = str(datetime.timedelta(seconds=int(duration)))
            report["file_info"]["recorded"] = datetime.datetime.fromtimestamp(first_packet_time).strftime('%Y-%m-%d %H:%M')

        internal = []
        external = []
        for ip in ips:
            if ip.startswith(('192.168.', '10.', '172.')):
                internal.append(ip)
            else:
                external.append(ip)
        
        report["metrics"]["internal_hosts"] = len(internal)
        report["metrics"]["external_hosts"] = len(external)
        report["metrics"]["unique_ips"] = len(ips)
        report["metrics"]["unique_domains"] = len(domains)
        report["protocols"] = dict(protocols.most_common(5))
        
        if cleartext_protocols:
            report["vulnerabilities"].append({
                "title": "CLEARTEXT PROTOCOLS",
                "count": protocols[cleartext_protocols[0]],
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

        for src, count in icmp_packets.items():
            if count > 50:
                report["suspicious_activities"].append({
                    "title": "POSSIBLE ICMP FLOOD",
                    "count": count,
                    "desc": f"Host {src} is sending a high volume of ICMP packets (Potential DoS/Scan)."
                })
                report["metrics"]["risk_score"] += 40
                break

        for src, count in syn_packets.items():
            if count > 100:
                report["suspicious_activities"].append({
                    "title": "SYN SCAN/FLOOD DETECTED",
                    "count": count,
                    "desc": f"Host {src} initiated {count} SYN requests. Characteristic of port scanning or SYN flood attack."
                })
                report["metrics"]["risk_score"] += 35
                break

        sorted_times = sorted(timeline_bins.keys())
        if sorted_times:
            report["timeline"]["timestamps"] = [datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in sorted_times]
            report["timeline"]["packet_counts"] = [timeline_bins[t] for t in sorted_times]
            report["timeline"]["byte_counts"] = [bytes_bins[t] for t in sorted_times]

        report["packet_log"] = packet_log

        report["metrics"]["risk_score"] = min(report["metrics"]["risk_score"], 100)

        return report

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("PCAP Engine Initialized")

        
