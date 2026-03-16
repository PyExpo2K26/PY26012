import sys
import os

# Add current directory to path to import engine
sys.path.append(os.getcwd())

from engine.pcap import analyze_pcap

def test_pcap():
    pcap_path = "test_ipv4frags.pcap"
    if not os.path.exists(pcap_path):
        print(f"Error: {pcap_path} not found")
        return

    print(f"Analyzing {pcap_path}...")
    
    import pyshark
    cap = pyshark.FileCapture(pcap_path, tshark_path=r"C:\Program Files\Wireshark\tshark.exe")
    for i, pkt in enumerate(cap):
        print(f"Packet {i}: {pkt.highest_layer}")
        for layer in pkt.layers:
            print(f"  Layer: {layer.layer_name}")
        if 'IP' in pkt:
            if hasattr(pkt.ip, 'flags_mf'):
                print(f"  IP MF Flag: {pkt.ip.flags_mf}")
            if hasattr(pkt.ip, 'frag_offset'):
                print(f"  IP Frag Offset: {pkt.ip.frag_offset}")
    cap.close()

    report = analyze_pcap(pcap_path)
    
    if "error" in report:
        print(f"Analysis failed: {report['error']}")
    else:
        print("Analysis successful!")
        print(f"Total Packets: {report['metrics']['total_packets']}")
        print(f"Protocols: {report['protocols']}")
        print(f"Vulnerabilities: {report['vulnerabilities']}")
        print(f"Suspicious Activities: {report['suspicious_activities']}")
        print(f"Risk Score: {report['metrics']['risk_score']}")

if __name__ == "__main__":
    test_pcap()
