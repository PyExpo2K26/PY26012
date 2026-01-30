# Digital Forgery Detection System with Forensics (Image, Video, Audio + Blockchain Evidence Hashing)


**EvidenX** is an intelligent digital forensics platform designed to uncover the hidden truth behind manipulated and AI-generated media. By combining advanced machine-learning forensic analysis with blockchain-secured evidence storage,EvidenX ensures every verification result is accurate, transparent, and tamper-proof. The system not only detects forgery but visually explains how, where, and when manipulation occurred. With its interactive cyber-forensic dashboard, users can analyze media effortlessly without technical expertise. EvidenX transforms digital verification into a legally reliable, future-ready solution for trust in the digital world.

## The main goal of this project is to:
     
     - Reliably detect forgery in digital media (images, videos, audio)
     - Clearly explain **how, where, and when** manipulation occurred
     - Guarantee evidence trustworthiness using blockchain
     - Deliver accurate, transparent, and legally admissible results
     - Build an ML-powered forensic system with non-repudiable report storage

## Technology Stack 

| Layer              | Main Technologies                     | Purpose                                      |
|--------------------|----------------------------------------|----------------------------------------------|
| Frontend           | HTML5 + CSS3 + Vanilla JS             | Responsive UI, drag-drop, real-time updates  |
| Backend            | FastAPI                               | Fast async API server                        |
| Machine Learning   | PyTorch + CNN models                  | Deepfake & manipulation detection            |
| Forensics          | EXIF tools, ELA, Copy-Move algos      | Classic forgery detection methods            |
| Blockchain         | Web3 + Sepolia + Infura + SHA-256     | Immutable evidence & report storage          |
| Communication      | JSON + Base64                         | File & data transfer                         |
| Security           | Bearer Token / API Key                | Endpoint protection                          |

## Core Features

### Multi-Media Forgery Detection
     - Images, videos, audio
     - Drag-and-drop upload
     - Major formats supported
     - Instant file validation
     - Auto type detection
     - Size & quality check
     - Preview before analysis

### Forensic Analysis Modules

#### EXIF Metadata Analysis
     - Reads camera, time, GPS, software traces
     - Detects timestamp mismatches, editing tools, spoofed metadata
     - Helps verify authenticity & timeline

#### Error Level Analysis (ELA)
     - Highlights compression differences
     - Creates heatmap of edited/spliced regions
     - Effective against copy-paste & subtle AI edits

#### Copy-Move Detection
     - Finds duplicated areas within the same image
     - Works even after scaling, rotation, minor edits
     - Exposes object removal / addition forgeries

#### CNN Classification
     - Deep learning model for deepfake detection
     - Spots unnatural faces, blending artifacts, frequency issues
     - Outputs confidence score (e.g. 92% fake)

### Progress Tracking UI
     - Real-time progress bars (EXIF · ELA · Copy-Move · CNN)
     - Estimated time remaining
     - Step-by-step logs & indicators

### Result Visualization
     - Overall forgery risk score
     - Per-module confidence
     - Final verdict (Authentic / Suspicious / Fake)
     - Risk badges (High / Medium / Low)

### Heatmap Visualization
     - Zoomable suspicious-region heatmap
     - Red = high risk
     - Yellow = possible issue
     - Green = likely original

### Blockchain Verification
     - Stores SHA-256 hash on-chain
     - Shows transaction hash
     - Provides chain-of-custody proof
     - Smart-contract verifiable

### Report Generation
     - PDF forensic report download
     - Shareable secure link
     - Analyst notes field
     - Digital signature support

### History Management
     - Saves past analyses
     - View / Download / Delete
     - Timestamped records

### Settings
     - Dark mode
     - Animation toggle
     - Email alerts
     - File size limit
     - Auto-cleanup

### API Integration
     - REST endpoints for analysis & reports
     - Bearer token auth
     - Python & JS SDK support

