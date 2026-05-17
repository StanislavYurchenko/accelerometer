# 📱 AI Activity Tracker (ONNX Runtime Web)

A web application for real-time human physical activity recognition using mobile phone sensors (accelerometer). The inference (execution) of machine learning models happens directly in the smartphone's browser using **ONNX Runtime Web**, entirely client-side without the need for a backend server.

## 🚀 Project Features
- **Sensor Reading**: Fetching raw acceleration data across three axes ($X, Y, Z$) using the browser's `DeviceMotionEvent` API.
- **Window Processing**: Slicing the dynamic data stream into sliding windows of 80 data points (~1.5–2 seconds of continuous movement).
- **Feature Extraction**: Calculating 15 mathematical features on the fly directly in JavaScript.
- **Classification**: Determining the current activity among 4 classes: `🧘‍♂️ Idle`, `🚶‍♂️ Walking`, `🧗‍♂️ Stairs`, `🏃‍♂️ Running`.
- **Debug Panel**: Real-time visualization of the feature array and an integrated button to quickly copy data to the clipboard for analysis.

---

## 🧠 Model Training (Python & Scikit-Learn)

The recognition model was developed in a Python environment using the following pipeline:

1. **Architecture**: An ensemble **Random Forest Classifier** algorithm was chosen for its high robustness to noise and extremely fast execution on mobile processors.
2. **Data Preprocessing**: **StandardScaler** (Z-score normalization) was used for feature scaling.
3. **Feature Engineering**: The original dataset contained raw accelerometer readings. For each time window (80 points), 15 statistical features were calculated:
   - Mean value (`mean`) for the $X, Y, Z$ axes.
   - Standard deviation (`std` / variance) for the $X, Y, Z$ axes.
   - Maximums (`max`) and minimums (`min`) for the $X, Y, Z$ axes.
   - `SMA` (Signal Magnitude Area) to evaluate the total signal energy.
   - Mean and standard deviation for the overall acceleration vector (Magnitude: $\sqrt{x^2 + y^2 + z^2}$).
4. **ONNX Conversion**: After training, the pipeline (`scaler` and `classifier`) was exported to `.onnx` format using the `skl2onnx` library.

---

## 🛠️ Technical Challenges & Bug Fixes (Web Integration)

Porting the ML model from Python to mobile JavaScript encountered several tricky development issues that were successfully resolved:

### 1. Motion Sensor Security & HTTPS
Modern mobile browsers (especially Chrome and Safari on iOS) completely block access to `devicemotion` on unsecured connections (`http://`). 
* **Solution**: The project is deployed on **GitHub Pages**, which automatically provides full encryption via `https://`, granting access to the system sensors.

### 2. Bypassing GitHub Pages Limits (ArrayBuffer Loading)
When trying to fetch binary `.onnx` files directly, the GitHub Pages server occasionally returned errors or incorrect file MIME types.
* **Solution**: The `initML()` logic was rewritten. Now, JavaScript fetches the models as raw bytes via `fetch().arrayBuffer()`, and ONNX is initialized directly from memory using `new Uint8Array(buffer)`. This ensures stable loading of models on any hosting provider.

### 3. Beating the ZipMap Bug in ONNX Runtime Web
The `skl2onnx` library adds a `ZipMap` operator to the Random Forest output graph by default to pack class probabilities into a dictionary. The mobile ONNX engine crashes with a critical error: `Can't access output tensor data on index 1... non-tensor typed value is not supported`.
* **Solution**: Strict output limitation was implemented during inference. The `session.run()` method is called with a specific array of expected output keys `['output_label']` (or `['label']`). The browser fetches only the clean predicted class tensor, completely ignoring the problematic probability map.

---

## 💡 Data Science Insight: The "Domain Shift" Effect

During live testing, a clear example of **Domain Shift** was discovered:
- The model **flawlessly recognizes Running**, as the amplitude of oscillations during a run easily breaks through the thresholds set by the decision trees.
- However, the model tends to classify normal, smooth **Walking** (when the phone is held in a relaxed hand) as **Idle**. 

*Why does this happen?* During walking, the human arm acts as a natural shock absorber, dampening the micro-impacts of feet hitting the ground. The live data turns out to be too "smooth" for a model that was trained on a laboratory dataset with a more rigid sensor attachment (e.g., in a pocket or strapped to a leg). This demonstrates a fundamental rule of Data Science: a model performs exactly as it was taught, and real-world deployment always requires calibration of data collection.

---

## 📂 Repository Structure
- `index.html` — Application UI with versioning and a metrics panel.
- `app.js` — The core engine: sensor handling, calculation of 15 features, and ONNX inference logic.
- `style.css` — UI styling in a Dark Cyberpunk aesthetic.
- `scaler.onnx` — Serialized data standardization model.
- `classifier.onnx` — Serialized Random Forest model.