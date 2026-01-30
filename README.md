# Digital Forgery Detection System with Forensics (Image, Video, Audio + Blockchain Evidence Hashing)


EvidenX is an intelligent digital forensics platform designed to uncover the hidden truth behind manipulated and AI-generated media. By combining advanced machine-learning forensic analysis with blockchain-secured evidence storage,EvidenX ensures every verification result is accurate, transparent, and tamper-proof. The system not only detects forgery but visually explains how, where, and when manipulation occurred. With its interactive cyber-forensic dashboard, users can analyze media effortlessly without technical expertise. EvidenX transforms digital verification into a legally reliable, future-ready solution for trust in the digital world.

The main goal of this project is to:
     • To identify digital media forgery reliably, explain the manipulation clearly to users, and ensure evidence trustworthiness using blockchain technology.
     •To uncover digital forgeries with accuracy, transparency, and legally reliable evidence preservation.
     •To develop a machine-learning-based forensic system capable of detecting media manipulation and securely preserving forensic reports on a blockchain to ensure non-repudiation and evidentiary integrity.

# Core Features (System Capabilities)

### Multi-Media Forgery Detection
        ◦ Supports images, videos, and audio uploads
        ◦ Drag-and-drop + file browser upload
        ◦ Accepts all major media formats
        ◦ Live file integrity check during upload
        ◦ Automatic media type detection
        ◦ Size and quality validation before processing
        ◦ Preview before submission
### Forensic Analysis Modules

#### EXIF Metadata Analysis
**What it does & why it matters in forensics / deepfake detection:**  
EXIF (Exchangeable Image File Format) is hidden metadata automatically embedded in photos by cameras and smartphones. It records camera model, date/time taken, GPS location, exposure settings, and editing software used.  
In **cyber forensics**, inconsistencies (e.g., mismatched timestamps, traces of editing software like Photoshop, or impossible GPS jumps) reveal tampering, metadata spoofing, or post-capture manipulation — helping verify authenticity and reconstruct event timelines.

#### Error Level Analysis (ELA)
**What it does & why it matters in forensics / deepfake detection:**  
ELA recompresses the image at a known JPEG quality level and compares compression "error" differences across regions. Edited, spliced, or AI-generated areas often show different compression artifacts because they were saved at another quality or originated from a different source.  
In **forgery & deepfake detection**, ELA produces visual heatmaps highlighting tampered zones (brighter = more suspicious), making it effective against splicing, copy-move forgeries, and subtle AI inconsistencies.

#### Copy-Move Detection
**What it does & why it matters in forensics / deepfake detection:**  
Copy-move forgery involves duplicating (copying) a region from the same image and pasting it elsewhere to hide/remove objects or fabricate elements.  
The module uses noise pattern comparison, scale-invariant matching, and duplication detection to identify identical or near-identical patches — even after rotation, scaling, or minor edits.  
In **cyber forensics**, it exposes cloning manipulations commonly used in image tampering and evidence alteration.

#### CNN Classification
**What it does & why it matters in forensics / deepfake detection:**  
Convolutional Neural Networks (CNNs) are deep learning models trained on large datasets of real vs. manipulated media to automatically learn subtle forgery patterns.  
For **deepfake detection**, the model analyzes face landmark inconsistencies (unnatural eye/mouth movements, blending edges), texture artifacts, frequency-domain anomalies, and ensemble predictions to output a confidence score (e.g., 92% fake).  
CNNs excel at catching sophisticated AI-generated fakes (GAN-based deepfakes) that traditional pixel-based methods often miss.

