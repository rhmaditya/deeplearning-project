"""
SCRIPT: Generate PDF Documentation with Sample Model Output
=============================================================
This script creates a comprehensive PDF documentation including:
1. Training vs Validation plots
2. Confusion matrices
3. Model comparison table
4. Sample predictions from CSV
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from datetime import datetime
import os

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PDF = "Model_Evaluation_Documentation_Complete.pdf"
MODEL_COMPARISON_CSV = "model_comparison.csv"
SAMPLE_OUTPUT_CSV = "sample_model_output.csv"

# Model files
MODELS = {
    "LSTM": {
        "path": "lstm_model.h5",
        "log": "lstm_training_log.csv"
    },
    "GRU": {
        "path": "gru_model.h5",
        "log": "gru_training_log.csv"
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_title_page(pdf):
    """Create title page"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    title_text = """
    DOKUMENTASI EVALUASI MODEL
    DETEKSI BERITA HOAX INDONESIA
    
    
    Tugas Besar Deep Learning
    Kelompok 5
    
    
    Institut Teknologi Sumatera
    2024
    
    
    Model yang Digunakan:
    • LSTM: Long Short-Term Memory
    • GRU: Gated Recurrent Unit
    
    Total Samples: 11,000+ berita
    Vocabulary Size: 10,000 tokens
    """
    
    ax.text(0.5, 0.5, title_text, 
            ha='center', va='center',
            fontsize=14, fontfamily='serif',
            transform=ax.transAxes)
    
    # Add date
    date_text = f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"
    ax.text(0.5, 0.05, date_text,
            ha='center', va='bottom',
            fontsize=10, style='italic',
            transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_training_history(log_file, model_name, pdf):
    """Plot training vs validation curves"""
    try:
        # Load training log
        history = pd.read_csv(log_file)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss
        ax1.plot(history['loss'], label='Training Loss', linewidth=2, color='#3498db')
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='#2ecc71')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plotted for {model_name}")
        
    except FileNotFoundError:
        print(f"⚠ Warning: {log_file} not found. Skipping training plot for {model_name}")
    except Exception as e:
        print(f"⚠ Error plotting training history for {model_name}: {str(e)}")

def plot_model_comparison(comparison_df, pdf):
    """Plot model comparison table and chart"""
    
    # Page 1: Table
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = comparison_df.reset_index()
    table_data.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
                          'Architecture', 'Tokenizer', 'Embedding']
    
    # Format numeric columns
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f"{float(x):.4f}")
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Highlight best model (first row)
    for j in range(len(table_data.columns)):
        table[(1, j)].set_facecolor('#2ecc71')
        table[(1, j)].set_text_props(weight='bold')
    
    plt.title("Model Comparison Table\nDeteksi Berita Hoax Indonesia",
              fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Page 2: Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            values = comparison_df[metric].astype(float)
            bars = ax.bar(x + i * width, values, width, 
                         label=metric.capitalize(), 
                         color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\nDeteksi Berita Hoax Indonesia',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df.index, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    print("✓ Model comparison plotted")

def create_sample_output_pages(sample_df, pdf):
    """Create detailed pages for sample model outputs"""
    
    print("\n   Creating sample output pages...")
    
    # Overview page with statistics
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Sample Model Output - Overview", 
                fontsize=16, fontweight='bold', y=0.97)
    
    # Calculate statistics
    total_samples = len(sample_df)
    correct = len(sample_df[sample_df['status'].str.contains('CORRECT')])
    incorrect = total_samples - correct
    fp = len(sample_df[sample_df['type'] == 'false_positive'])
    fn = len(sample_df[sample_df['type'] == 'false_negative'])
    
    # Create text summary
    summary_text = f"""
{'='*70}
STATISTICS SUMMARY
{'='*70}

Total Samples:           {total_samples}
Correct Predictions:     {correct} ({correct/total_samples*100:.1f}%)
Incorrect Predictions:   {incorrect} ({incorrect/total_samples*100:.1f}%)

Error Breakdown:
  • False Positives (Real → Fake):  {fp} ({fp/total_samples*100:.1f}%)
  • False Negatives (Fake → Real):  {fn} ({fn/total_samples*100:.1f}%)

Confidence Distribution:
  • Very High (>90%):  {len(sample_df[sample_df['confidence'] > 0.9])} samples
  • High (70-90%):     {len(sample_df[(sample_df['confidence'] > 0.7) & (sample_df['confidence'] <= 0.9)])} samples
  • Medium (50-70%):   {len(sample_df[(sample_df['confidence'] > 0.5) & (sample_df['confidence'] <= 0.7)])} samples
  • Low (<50%):        {len(sample_df[sample_df['confidence'] <= 0.5])} samples

Sample Types:
"""
    
    # Add type distribution
    type_counts = sample_df['type'].value_counts()
    for type_name, count in type_counts.items():
        summary_text += f"  • {type_name}: {count}\n"
    
    summary_text += f"\n{'='*70}"
    
    fig.text(0.1, 0.9, summary_text,
            fontsize=10, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Create detailed page for each sample
    for idx, row in sample_df.iterrows():
        create_detailed_sample_page(row, idx + 1, total_samples, pdf)
    
    print(f"✓ Created {total_samples} sample output pages")

def create_detailed_sample_page(row, sample_num, total_samples, pdf):
    """Create detailed page for single sample output"""
    
    fig = plt.figure(figsize=(8.5, 11))
    
    # Determine status and color
    is_correct = 'CORRECT' in row['status']
    status_color = 'darkgreen' if is_correct else 'darkred'
    bg_color = 'lightgreen' if is_correct else 'lightyellow'
    
    # Determine confidence level
    conf = row['confidence']
    if conf > 0.9:
        conf_level = "VERY HIGH"
    elif conf > 0.7:
        conf_level = "HIGH"
    elif conf > 0.5:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"
    
    # Determine error type
    error_info = ""
    if not is_correct:
        if row['type'] == 'false_positive':
            error_info = "\n❌ ERROR TYPE: FALSE POSITIVE\n   (Real news predicted as Fake)"
        elif row['type'] == 'false_negative':
            error_info = "\n❌ ERROR TYPE: FALSE NEGATIVE\n   (Fake news predicted as Real)"
        elif row['type'] == 'satire_missed':
            error_info = "\n❌ ERROR TYPE: SATIRE MISCLASSIFICATION\n   (Satire content not detected)"
    
    # Create title with status emoji
    status_emoji = "✅" if is_correct else "❌"
    title_text = f"{status_emoji} SAMPLE #{sample_num}/{total_samples}"
    
    fig.text(0.5, 0.96, title_text,
            fontsize=14, fontweight='bold',
            ha='center', color=status_color)
    
    # Main content
    content_text = f"""
{'='*70}
INPUT TEXT
{'='*70}
{row['text']}

PREDICTION RESULTS

True Label:          {row['true_label']}
Predicted Label:     {row['predicted_label']}
Raw Probability:     {row['probability']:.4f}
                     ({row['probability']*100:.2f}% chance of being FAKE)

Confidence:          {conf:.4f} ({conf*100:.2f}%)
Confidence Level:    {conf_level}

Status:              {row['status']}{error_info}

ANALYSIS
"""
    
 # Tambahkan analisis spesifik berdasarkan tipe
    if row['type'] == 'correct_high_conf':
        content_text += "\n✓ Prediksi benar dengan confidence tinggi"
        content_text += "\n✓ Model berkinerja sesuai ekspektasi"
        
    elif row['type'] == 'medical_hoax':
        content_text += "\n⚠️  Misinformasi medis terdeteksi"
        content_text += "\n✓ Model berhasil mengidentifikasi pseudosains"
        
    elif row['type'] == 'political_hoax':
        content_text += "\n⚠️  Manipulasi politik terdeteksi"
        content_text += "\n✓ Indikator teori konspirasi ditemukan"
        
    elif row['type'] == 'false_positive':
        content_text += "\n❌ Model salah menandai berita asli sebagai hoax"
        content_text += "\n⚠️  Kemungkinan disebabkan oleh judul sensasional"
        content_text += "\n💡 Rekomendasi: Tambahkan layer verifikasi sumber"
        
    elif row['type'] == 'false_negative':
        content_text += "\n❌ Hoax sophisticated lolos dari deteksi"
        content_text += "\n⚠️  Struktur formal menyerupai berita asli"
        content_text += "\n💡 Rekomendasi: Integrasikan API fact-checking"
        
    elif row['type'] == 'borderline':
        content_text += "\n⚠️  Prediksi dengan confidence rendah"
        content_text += "\n⚠️  Probabilitas mendekati threshold keputusan (0.5)"
        content_text += "\n💡 Rekomendasi: Diperlukan verifikasi manual"
        
    elif row['type'] == 'satire_missed':
        content_text += "\n❌ Konten satire/parodi tidak terdeteksi"
        content_text += "\n⚠️  Model belum memiliki deteksi sarkasme"
        content_text += "\n💡 Rekomendasi: Tambahkan database website satire"
        
    elif row['type'] == 'opinion_piece':
        content_text += "\n✓ Artikel opini berhasil diklasifikasikan dengan benar"
        content_text += "\n⚠️  Confidence rendah karena konten subjektif"
        
    elif row['type'] == 'tech_news':
        content_text += "\n✓ Berita teknologi berhasil diidentifikasi dengan akurat"
        content_text += "\n✓ Terminologi formal terkenali dengan baik"
    
    content_text += f"\n\n{'='*70}"
    
    # Display content
    fig.text(0.05, 0.90, content_text,
            fontsize=8, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
    
    # Add footer
    footer_text = f"Model: GRU  | Page {sample_num + 3}/{total_samples + 3}"
    fig.text(0.5, 0.02, footer_text,
            fontsize=8, style='italic',
            ha='center', va='bottom')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN SCRIPT
# ============================================================

def generate_pdf_documentation():
    """Main function to generate PDF documentation"""
    
    print("="*60)
    print("GENERATING PDF DOCUMENTATION WITH SAMPLE OUTPUT")
    print("="*60)
    
    # Load model comparison
    print("\n1. Loading model comparison...")
    try:
        model_comparison = pd.read_csv(MODEL_COMPARISON_CSV, index_col=0)
        print(f"✓ Model comparison loaded: {len(model_comparison)} models")
    except Exception as e:
        print(f"✗ Error loading model comparison: {str(e)}")
        return
    
    # Load sample output
    print("\n2. Loading sample model output...")
    try:
        sample_output = pd.read_csv(SAMPLE_OUTPUT_CSV)
        print(f"✓ Sample output loaded: {len(sample_output)} samples")
    except Exception as e:
        print(f"✗ Error loading sample output: {str(e)}")
        return
    
    # Create PDF
    print(f"\n3. Creating PDF: {OUTPUT_PDF}")
    with PdfPages(OUTPUT_PDF) as pdf:
        
        # Title page
        print("\n   Creating title page...")
        create_title_page(pdf)
        
        # Model comparison
        print("\n   Creating model comparison...")
        plot_model_comparison(model_comparison, pdf)
        
        # Training history for each model
        for model_name, model_info in MODELS.items():
            print(f"\n   Processing {model_name}...")
            
            # Plot training history
            if 'log' in model_info:
                plot_training_history(model_info['log'], model_name, pdf)
        
        # Sample output pages
        create_sample_output_pages(sample_output, pdf)
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Model Evaluation Documentation - Deteksi Berita Hoax Indonesia'
        d['Author'] = 'Kelompok 5 Deep Learning'
        d['Subject'] = 'Deep Learning Model Evaluation with Sample Predictions'
        d['Keywords'] = 'Hoax Detection, Deep Learning, RNN, LSTM, GRU, Sample Output'
        d['CreationDate'] = datetime.now()
    
    print("\n" + "="*60)
    print(f"✓ PDF DOCUMENTATION CREATED: {OUTPUT_PDF}")
    print("="*60)
    print(f"\nPDF contains:")
    print("  • Title page")
    print("  • Model comparison (table + chart)")
    print("  • Training history plots (3 models)")
    print(f"  • Sample output overview")
    print(f"  • Detailed sample predictions ({len(sample_output)} samples)")
    print(f"\nTotal pages: ~{3 + 2 + 3 + 1 + len(sample_output)} pages")

# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    generate_pdf_documentation()