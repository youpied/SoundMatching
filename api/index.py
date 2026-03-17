from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
import io
import os
import base64
import matplotlib.pyplot as plt

app = FastAPI()

# 31バンドEQ周波数
eq_freqs = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])

def smooth_spectrum(spec):
    return uniform_filter1d(spec, size=15)

def calculate_eq_logic(clean, recorded, sr):
    n_fft = 4096
    
    clean_spec = np.abs(librosa.stft(clean, n_fft=n_fft))
    rec_spec = np.abs(librosa.stft(recorded, n_fft=n_fft))

    clean_avg = np.mean(clean_spec, axis=1)
    rec_avg = np.mean(rec_spec, axis=1)

    clean_db = librosa.amplitude_to_db(clean_avg)
    rec_db = librosa.amplitude_to_db(rec_avg)

    clean_db = smooth_spectrum(clean_db)
    rec_db = smooth_spectrum(rec_db)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    eq_values = []

    for f in eq_freqs:
        low = f / np.sqrt(2)
        high = f * np.sqrt(2)
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) > 0:
            diff = np.mean(rec_db[idx] - clean_db[idx])
        else:
            diff = 0
        eq_values.append(diff)

    eq_values = np.array(eq_values)
    eq_values *= 0.7
    eq_values -= np.mean(eq_values)
    eq_values = np.clip(eq_values, -20, 20)
    
    return np.round(eq_values).astype(int)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sound Match EQ Generator</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0c0c0e;
            --card-color: #1a1a1e;
            --accent-color: #0071e3;
            --text-color: #f5f5f7;
            --text-secondary: #86868b;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            width: 90%;
            max-width: 800px;
            background: var(--card-color);
            padding: 40px;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 30px;
            text-align: center;
            background: linear-gradient(135deg, #fff, #888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .upload-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .upload-card {
            background: rgba(255,255,255,0.03);
            border: 2px dashed rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
        }

        .upload-card:hover {
            border-color: var(--accent-color);
            background: rgba(0,113,227,0.05);
        }

        label {
            cursor: pointer;
            display: block;
        }

        input[type="file"] {
            display: none;
        }

        .file-status {
            margin-top: 10px;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .btn-generate {
            display: block;
            width: 100%;
            padding: 15px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-bottom: 20px;
        }

        .btn-generate:hover {
            transform: scale(1.02);
            filter: brightness(1.1);
        }

        .btn-generate:active {
            transform: scale(0.98);
        }

        #result {
            margin-top: 30px;
            display: none;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            padding-right: 10px;
        }

        .eq-item {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-size: 0.9rem;
        }

        .eq-freq { color: var(--text-secondary); }
        .eq-val { font-weight: 600; color: var(--accent-color); }

        #graph-container {
            margin-top: 30px;
            text-align: center;
        }

        #graph-img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }

        .loading {
            text-align: center;
            display: none;
            color: var(--text-secondary);
            margin-top: 20px;
        }

        .loading::after {
            content: "";
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid var(--accent-color);
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
            vertical-align: middle;
            margin-left: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sound Match EQ</h1>
        
        <div class="upload-section">
            <div class="upload-card">
                <label for="clean_file">
                    <strong>Original Audio</strong>
                    <div class="file-status" id="clean-status">Not selected</div>
                </label>
                <input type="file" id="clean_file" accept=".wav,.mp3,.flac,.m4a">
            </div>
            <div class="upload-card">
                <label for="recorded_file">
                    <strong>Recorded Audio</strong>
                    <div class="file-status" id="recorded-status">Not selected</div>
                </label>
                <input type="file" id="recorded_file" accept=".wav,.mp3,.flac,.m4a">
            </div>
        </div>

        <button class="btn-generate" onclick="generateEQ()">Generate EQ</button>

        <div class="loading" id="loader">Processing audio...</div>

        <div id="result">
            <div id="graph-container">
                <img id="graph-img" src="" alt="EQ Graph">
            </div>
            <h3>EQ Settings</h3>
            <div class="result-grid" id="eq-result-grid"></div>
        </div>
    </div>

    <script>
        document.getElementById('clean_file').onchange = e => {
            document.getElementById('clean-status').innerText = e.target.files[0].name;
        };
        document.getElementById('recorded_file').onchange = e => {
            document.getElementById('recorded-status').innerText = e.target.files[0].name;
        };

        async function generateEQ() {
            const cleanFile = document.getElementById('clean_file').files[0];
            const recordedFile = document.getElementById('recorded_file').files[0];

            if (!cleanFile || !recordedFile) {
                alert("Please select both files.");
                return;
            }

            const formData = new FormData();
            formData.append('clean', cleanFile);
            formData.append('recorded', recordedFile);

            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            try {
                const response = await fetch('/calculate', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Processing failed");

                const data = await response.json();
                
                // Display results
                const grid = document.getElementById('eq-result-grid');
                grid.innerHTML = '';
                data.eq_values.forEach((v, i) => {
                    const freq = data.freqs[i];
                    const div = document.createElement('div');
                    div.className = 'eq-item';
                    div.innerHTML = `<div class="eq-freq">${freq}Hz</div><div class="eq-val">${v > 0 ? '+' : ''}${v}</div>`;
                    grid.appendChild(div);
                });

                document.getElementById('graph-img').src = 'data:image/png;base64,' + data.graph;
                document.getElementById('result').style.display = 'block';

            } catch (err) {
                alert(err.message);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """

@app.post("/calculate")
async def calculate_eq(clean: UploadFile = File(...), recorded: UploadFile = File(...)):
    try:
        # Load audio files
        clean_bytes = await clean.read()
        recorded_bytes = await recorded.read()
        
        # We need soundfile or audioread to load from bytes
        # librosa.load can take a file object
        clean_io = io.BytesIO(clean_bytes)
        recorded_io = io.BytesIO(recorded_bytes)
        
        y_clean, sr_clean = librosa.load(clean_io, sr=None)
        y_recorded, sr_recorded = librosa.load(recorded_io, sr=None)
        
        if sr_clean != sr_recorded:
            # Resample recorded to clean sr if they are different
            y_recorded = librosa.resample(y_recorded, orig_sr=sr_recorded, target_sr=sr_clean)
            sr_recorded = sr_clean

        eq_values = calculate_eq_logic(y_clean, y_recorded, sr_clean)
        
        # Generate Plot
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.plot(eq_freqs, eq_values, marker='o', color='#0071e3', linewidth=2)
        plt.xscale('log')
        plt.xticks(eq_freqs[::3], [f"{f}" for f in eq_freqs[::3]])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Generated EQ Curve')
        plt.grid(True, alpha=0.2)
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "eq_values": eq_values.tolist(),
            "freqs": eq_freqs.tolist(),
            "graph": graph_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
