import email
from email import policy
import re
import requests
import socket

def extract_ip_from_received(received_headers):
    """
    Extracts the sender IP from Received headers.
    Usually the last (bottom-most) Received header contains the origin IP.
    """
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    ips = []
    
    for header in received_headers:
        matches = re.findall(ip_pattern, header)
        for ip in matches:
            # Basic check to avoid private IPs if possible, though first hop might be internal
            if not ip.startswith(('10.', '192.168.', '172.16.', '127.0.0.')):
                ips.append(ip)
            else:
                # Keep it as a backup if no public IP is found
                ips.append(ip)
                
    # In forensics, the bottom-most 'Received' header is often the most interesting (origin)
    # But sometimes the top-most is the last hop. We'll return the list and let the UI decide or pick the last.
    return ips[-1] if ips else None

def get_geolocation(ip):
    """
    Fetches geolocation data for an IP address using ip-api.com (free).
    """
    if not ip or ip.startswith(('10.', '192.168.', '127.')):
        return None
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Geolocation error: {e}")
    return None

def analyze_email_headers(file_path):
    """
    Parses .eml file and extracts forensic information.
    """
    with open(file_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
        
    report = {
        "Subject": msg.get('Subject', 'No Subject'),
        "From": msg.get('From', 'Unknown'),
        "To": msg.get('To', 'Unknown'),
        "Date": msg.get('Date', 'Unknown'),
        "Message-ID": msg.get('Message-ID', 'Unknown'),
        "Return-Path": msg.get('Return-Path', 'Unknown'),
        "Authentication-Results": msg.get('Authentication-Results', 'None'),
    }
    
    # Extract Authenticity Status (SPF, DKIM, DMARC)
    auth_results = report["Authentication-Results"]
    report["SPF"] = "Unknown"
    report["DKIM"] = "Unknown"
    report["DMARC"] = "Unknown"
    
    if "spf=" in auth_results.lower():
        match = re.search(r'spf=(\w+)', auth_results.lower())
        if match: report["SPF"] = match.group(1).upper()
        
    if "dkim=" in auth_results.lower():
        match = re.search(r'dkim=(\w+)', auth_results.lower())
        if match: report["DKIM"] = match.group(1).upper()

    if "dmarc=" in auth_results.lower():
        match = re.search(r'dmarc=(\w+)', auth_results.lower())
        if match: report["DMARC"] = match.group(1).upper()
        
    # Get Received headers for IP analysis
    received = msg.get_all('Received', [])
    report["Hops"] = received
    sender_ip = extract_ip_from_received(received)
    report["Sender_IP"] = sender_ip
    
    # Geolocation
    geo = get_geolocation(sender_ip)
    report["Geolocation"] = geo
    
    # Forensic Risk Scoring (Professional Algorithm)
    risk_score = 0
    reasons = []
    
    # 1. SPF Score (Max 30)
    spf_status = report["SPF"].upper()
    if spf_status == "FAIL":
        risk_score += 30
        reasons.append("CRITICAL: SPF check failed (Identity Spoofing)")
    elif spf_status == "SOFTFAIL":
        risk_score += 20
        reasons.append("WARNING: SPF soft-fail (Suspicious Origin)")
    elif spf_status in ["UNKNOWN", "NONE", ""]:
        risk_score += 10
        reasons.append("INFO: No SPF record found")
        
    # 2. DKIM Score (Max 40)
    dkim_status = report["DKIM"].upper()
    if dkim_status == "FAIL":
        risk_score += 40
        reasons.append("CRITICAL: DKIM signature invalid (Modified Content)")
    elif dkim_status in ["UNKNOWN", "NONE", ""]:
        risk_score += 15
        reasons.append("INFO: No DKIM signature present")
        
    # 3. DMARC Score (Max 20)
    dmarc_status = report["DMARC"].upper()
    if dmarc_status == "FAIL":
        risk_score += 20
        reasons.append("WARNING: DMARC policy violation")
    elif dmarc_status in ["UNKNOWN", "NONE", ""]:
        risk_score += 5
        reasons.append("INFO: No DMARC policy found")
        
    # 4. Domain Spoofing Check (Max 40)
    try:
        from_header = report["From"]
        from_match = re.search(r'[\w\.-]+@[\w\.-]+', from_header)
        if from_match:
            from_domain = from_match.group(0).split('@')[-1].lower()
            
            # Check Return-Path
            rp_header = report["Return-Path"]
            if rp_header and rp_header != 'Unknown':
                rp_match = re.search(r'[\w\.-]+@[\w\.-]+', rp_header)
                if rp_match:
                    rp_domain = rp_match.group(0).split('@')[-1].lower()
                    if from_domain != rp_domain:
                        risk_score += 40
                        reasons.append(f"ALARM: Envelope Domain Mismatch (From: {from_domain} vs Return-Path: {rp_domain})")
            
            # Check Reply-To if different
            reply_to = msg.get('Reply-To')
            if reply_to:
                rt_match = re.search(r'[\w\.-]+@[\w\.-]+', reply_to)
                if rt_match:
                    rt_domain = rt_match.group(0).split('@')[-1].lower()
                    if from_domain != rt_domain:
                        risk_score += 15
                        reasons.append("WARNING: Reply-To domain differs from From domain")
    except Exception as e:
        print(f"Domain check error: {e}")

    report["Risk_Score"] = min(risk_score, 100)
    report["Risk_Level"] = "High" if report["Risk_Score"] >= 70 else "Medium" if report["Risk_Score"] >= 30 else "Low"
    report["Risk_Reasons"] = reasons
    
    return report
