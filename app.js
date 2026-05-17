const startBtn = document.getElementById('startBtn');
const statusDiv = document.getElementById('status');
const progressSpan = document.getElementById('windowProgress');
const currentMagSpan = document.getElementById('currentMag');
const featuresDebug = document.getElementById('featuresDebug');

let scalerSession = null;
let classifierSession = null;

let windowX = [];
let windowY = [];
let windowZ = [];
const WINDOW_SIZE = 80;

const activityClasses = {
    0: "🧘‍♂️ Rest",
    1: "🚶‍♂️ Walking",
    2: "🧗‍♂️ Stairs",
    3: "🏃‍♂️ Running"
};

// Load ONNX sessions
async function initML() {
    try {
        statusDiv.innerText = "Initializing AI...";
        scalerSession = await ort.InferenceSession.create('scaler.onnx');
        classifierSession = await ort.InferenceSession.create('classifier.onnx');
        statusDiv.innerText = "Ready to Track";
    } catch (e) {
        statusDiv.innerText = "ONNX Load Error: " + e.message;
        console.error(e);
    }
}

initML();

startBtn.addEventListener('click', async () => {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
            const permissionState = await DeviceMotionEvent.requestPermission();
            if (permissionState === 'granted') {
                startSensorTracking();
            } else {
                alert('Sensor access denied!');
            }
        } catch (error) {
            console.error(error);
        }
    } else {
        startSensorTracking();
    }
});

function startSensorTracking() {
    startBtn.style.display = 'none';
    statusDiv.innerText = "Tracking...";
    
    window.addEventListener('devicemotion', (event) => {
        let x = event.accelerationIncludingGravity.x || 0;
        let y = event.accelerationIncludingGravity.y || 0;
        let z = event.accelerationIncludingGravity.z || 0;

        let currentMag = Math.sqrt(x*x + y*y + z*z);
        currentMagSpan.innerText = currentMag.toFixed(2);

        windowX.push(x);
        windowY.push(y);
        windowZ.push(z);
        
        progressSpan.innerText = windowX.length;

        if (windowX.length >= WINDOW_SIZE) {
            runModelInference(windowX, windowY, windowZ);
            windowX = [];
            windowY = [];
            windowZ = [];
        }
    });
}

// Extract 15 features matching Python logic
function extractFeatures(X, Y, Z) {
    const getMean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const getStd = (arr, mean) => Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length);
    
    const x_mean = getMean(X);
    const y_mean = getMean(Y);
    const z_mean = getMean(Z);

    const x_std = getStd(X, x_mean);
    const y_std = getStd(Y, y_mean);
    const z_std = getStd(Z, z_mean);

    const x_max = Math.max(...X);
    const y_max = Math.max(...Y);
    const z_max = Math.max(...Z);

    const x_min = Math.min(...X);
    const y_min = Math.min(...Y);
    const z_min = Math.min(...Z);

    let smaSum = 0;
    for (let i = 0; i < WINDOW_SIZE; i++) {
        smaSum += Math.abs(X[i]) + Math.abs(Y[i]) + Math.abs(Z[i]);
    }
    const sma = smaSum / WINDOW_SIZE;

    let magnitudes = [];
    for (let i = 0; i < WINDOW_SIZE; i++) {
        magnitudes.push(Math.sqrt(X[i]**2 + Y[i]**2 + Z[i]**2));
    }
    const mag_mean = getMean(magnitudes);
    const mag_std = getStd(magnitudes, mag_mean);

    return [
        x_mean, y_mean, z_mean,
        x_std, y_std, z_std,
        x_max, y_max, z_max,
        x_min, y_min, z_min,
        sma,
        mag_mean, mag_std
    ];
}

async function runModelInference(X, Y, Z) {
    if (!scalerSession || !classifierSession) return;

    const features = extractFeatures(X, Y, Z);
    featuresDebug.innerText = JSON.stringify(features.map(f => f.toFixed(3)));

    const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 15]);

    try {
        // Step A: Standardization via scaler.onnx
        const scalerResults = await scalerSession.run({ float_input: inputTensor });
        const scaledFeatures = scalerResults.variable;

        // Step B: Classification via classifier.onnx
        const classifierResults = await classifierSession.run({ float_input: scaledFeatures });
        const predictedClass = classifierResults.label.data[0];

        statusDiv.innerText = activityClasses[predictedClass] || "Unknown";

    } catch (err) {
        console.error("ONNX Inference Error:", err);
        statusDiv.innerText = "AI Run Error: " + err.message;
    }
}