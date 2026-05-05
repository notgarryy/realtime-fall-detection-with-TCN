# Real-Time Fall Detection System with TCN

A real-time human fall detection system using **FMCW radar** and **Temporal Convolutional Networks (TCN)**, a deep learning architecture designed for spatiotemporal feature learning in human activity recognition.

> **Published Research**: This system is based on peer-reviewed work published in **IEEE Access (Q1)**, 2026.
> [View Paper](https://doi.org/10.1109/ACCESS.2026.3676850)
>
> **Related Work**: A TTSNet-based variant of this system was submitted in *Advance Sustainable Science, Engineering and Technology* and currently is under revision.

---

## Overview

This system processes live radar point cloud data from a **Texas Instruments IWR6843AOPEVM** sensor and classifies human activities in real time, with a primary focus on fall detection.

Overview of the proposed method:
![Proposed method](images/ProposedMethod.png)

---

## Features

- Real-time human activity recognition and fall detection
- TCN (Temporal Convolutional Network) for spatiotemporal feature extraction
- Visual real-time output with activity classification display
- Tested on Texas Instruments IWR6843AOPEVM radar

---

## Activity Classes

| Class | Description |
|---|---|
| Berdiri | Standing |
| Duduk | Sitting |
| Bungkuk | Bending |
| Jatuh | **Fall**  |

---

## Results

| Metric | Score |
|---|---|
| Model Accuracy | **99.82%** |
| Real-Time Average Accuracy | **87.78%** |

> **Note**: The gap between model accuracy and real-time accuracy is expected in radar-based HAR systems due to environmental variability, sensor noise, and the inherent difficulty of capturing consistent point cloud data in live conditions.

---

## Model Architecture

The TCN model captures spatiotemporal features from radar point cloud sequences using:
- **Temporal Convolutional Networks** for sequential activity recognition
- **FMCW radar point cloud data** as input — x, y, z coordinates, velocities, and accelerations
- **Sliding window** approach for continuous real-time inference

---

## Requirements

- Windows OS (tested on Windows 10)
- Python 3.10.0
- Texas Instruments IWR6843AOPEVM radar sensor

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/notgarryy/real-time-fall-detection-system-with-TCN.git
   cd real-time-fall-detection-system-with-TCN
   ```

2. **Install Python 3.10.0**
   Download from the [official Python website](https://www.python.org/downloads/release/python-3100/)

3. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Connect the radar sensor**
   Connect the IWR6843AOPEVM to your machine via USB before running.

---

## Usage

```bash
python main.py
```

---

## Publication

If you use this work, please cite:

```
Garry Nelson et al., "Spatiotemporal Feature Learning for Real-Time Human 
Fall Detection System Using TCNs and FMCW Radar Point Clouds," 
IEEE Access, 2026. DOI: 10.1109/ACCESS.2026.3676850
```

---

## Related Repository

- [TTSNet-based Fall Detection](https://github.com/notgarryy/real-time-fall-detection-system-with-TTSNet)

---

## Author

**Garry Nelson**
Electrical Engineering, Telkom University — Bandung, Indonesia

[GitHub](https://github.com/notgarryy) | [LinkedIn](https://www.linkedin.com/in/garry-nelson-889834277/)

---

*Developed as part of a bachelor's thesis at Telkom University, Bandung, Indonesia.*