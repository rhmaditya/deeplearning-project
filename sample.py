"""Generate sample model output untuk dokumentasi"""

import pandas as pd

# Contoh output 10 prediksi
sample_outputs = [
    {
        "text": "VIRAL!!! Vaksin COVID Mengandung Chip 5G...",
        "true_label": "FAKE",
        "predicted_label": "FAKE",
        "probability": 0.9823,
        "confidence": 0.9823,
        "status": "CORRECT",
        "type": "correct_high_conf"
    },
    {
        "text": "Presiden Joko Widodo mengumumkan kebijakan baru...",
        "true_label": "REAL",
        "predicted_label": "REAL",
        "probability": 0.0543,
        "confidence": 0.9457,
        "status": "CORRECT",
        "type": "correct_high_conf"
    },
    {
        "text": "BREAKING: Gempa Dahsyat Guncang Jakarta...",
        "true_label": "REAL",
        "predicted_label": "FAKE",
        "probability": 0.7234,
        "confidence": 0.7234,
        "status": "INCORRECT (FP)",
        "type": "false_positive"
    },
    {
        "text": "Menurut penelitian Harvard, air kelapa sembuhkan kanker...",
        "true_label": "FAKE",
        "predicted_label": "REAL",
        "probability": 0.3421,
        "confidence": 0.6579,
        "status": "INCORRECT (FN)",
        "type": "false_negative"
    },
    {
        "text": "Apple dikabarkan akan menghentikan iPhone 15...",
        "true_label": "FAKE",
        "predicted_label": "FAKE",
        "probability": 0.5789,
        "confidence": 0.5789,
        "status": "CORRECT (Borderline)",
        "type": "borderline"
    },
    {
        "text": "Pemerintah umumkan WFH permanen untuk PNS...",
        "true_label": "FAKE",
        "predicted_label": "REAL",
        "probability": 0.4123,
        "confidence": 0.5877,
        "status": "INCORRECT (Satire)",
        "type": "satire_missed"
    },
    {
        "text": "Bocoran! Calon Presiden X punya rekening offshore...",
        "true_label": "FAKE",
        "predicted_label": "FAKE",
        "probability": 0.9567,
        "confidence": 0.9567,
        "status": "CORRECT",
        "type": "political_hoax"
    },
    {
        "text": "Kebijakan PPN dinilai tidak tepat sasaran...",
        "true_label": "REAL",
        "predicted_label": "REAL",
        "probability": 0.3892,
        "confidence": 0.6108,
        "status": "CORRECT (Low Conf)",
        "type": "opinion_piece"
    },
    {
        "text": "Dokter Wuhan: COVID sembuh dengan air jahe...",
        "true_label": "FAKE",
        "predicted_label": "FAKE",
        "probability": 0.9912,
        "confidence": 0.9912,
        "status": "CORRECT",
        "type": "medical_hoax"
    },
    {
        "text": "Google umumkan Gemini 2.0 dengan AI multimodal...",
        "true_label": "REAL",
        "predicted_label": "REAL",
        "probability": 0.0234,
        "confidence": 0.9766,
        "status": "CORRECT",
        "type": "tech_news"
    }
]

# Convert to DataFrame
df_output = pd.DataFrame(sample_outputs)

# Save to CSV
df_output.to_csv("sample_model_output.csv", index=False)

print("✓ Sample output created: sample_model_output.csv")
print(f"Total samples: {len(sample_outputs)}")
print("\nOutput preview:")
print(df_output[['predicted_label', 'probability', 'status']].head())