import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QCheckBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import time
import json
from io import BytesIO

# Median values for quantization (32 groups)
MEDIAN_VALUES = [
    4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92,
    100, 108, 116, 124, 132, 140, 148, 156, 164, 172,
    180, 188, 196, 204, 212, 220, 228, 236, 244, 252
]

# Create mappings
MEDIAN_TO_INDEX = {median: idx for idx, median in enumerate(MEDIAN_VALUES)}
INDEX_TO_MEDIAN = {idx: median for idx, median in enumerate(MEDIAN_VALUES)}

class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, process_func, *args):
        super().__init__()
        self.process_func = process_func
        self.args = args

    def run(self):
        try:
            result = self.process_func(*self.args)
            self.processing_complete.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

def apply_median_quantization(img_array, progress_callback=None):
    """Quantize image to nearest median value (32 levels)."""
    quantized = np.zeros_like(img_array)
    total_pixels = img_array.size
    processed = 0
    
    for i, median in enumerate(MEDIAN_VALUES):
        lower = median - 4
        upper = median + 3
        mask = (img_array >= lower) & (img_array <= upper)
        quantized[mask] = median
        processed += np.sum(mask)
        
        if progress_callback and i % 4 == 0:  # Update progress every 4 steps
            progress_callback(int(processed / total_pixels * 100))
    
    return quantized

def reduce_to_5bit(quantized_img):
    """Convert median values to 5-bit indices (0-31) using numpy vectorization."""
    return np.vectorize(MEDIAN_TO_INDEX.get)(quantized_img)

def pack_5bit_to_bytes(five_bit_img):
    """Optimized packing of 5-bit indices into bytes using numpy bit operations."""
    flat = five_bit_img.flatten()
    # Pad with zeros if length not divisible by 8
    pad_length = (8 - (len(flat) % 8)) % 8
    padded = np.pad(flat, (0, pad_length), 'constant')
    
    # Reshape to group 8 values (40 bits) per row
    grouped = padded.reshape(-1, 8)
    
    # Convert each group to 5 bytes (40 bits)
    packed = np.zeros((grouped.shape[0], 5), dtype=np.uint8)
    for i in range(5):
        packed[:, i] = np.left_shift(grouped[:, i*8//5], (i*8) % 5) | \
                      np.right_shift(grouped[:, i*8//5 + 1], 5 - (i*8) % 5)
    
    return packed.tobytes(), five_bit_img.shape, pad_length

def unpack_bytes_to_5bit(packed_data, original_shape, pad_length):
    """Optimized unpacking of bytes back to 5-bit indices."""
    packed_arr = np.frombuffer(packed_data, dtype=np.uint8)
    packed_arr = packed_arr.reshape(-1, 5)
    
    # Reconstruct 5-bit values
    unpacked = np.zeros((packed_arr.shape[0], 8), dtype=np.uint8)
    for i in range(5):
        shift = (i*8) % 5
        unpacked[:, i*8//5] |= np.right_shift(packed_arr[:, i], shift)
        if i*8//5 + 1 < 8:
            unpacked[:, i*8//5 + 1] |= np.left_shift(packed_arr[:, i], 5 - shift)
    
    # Remove padding and reshape
    total_values = original_shape[0] * original_shape[1]
    return unpacked.flatten()[:total_values].reshape(original_shape)

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.processed_img = None
        self.packed_data = None
        self.original_shape = None
        self.is_color = False
        self.convert_to_grayscale = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Advanced Image Quantizer")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; font-size: 16px; margin: 10px; }
            QPushButton {
                background-color: #4CAF50; color: white; border-radius: 5px;
                padding: 8px; font-size: 14px;
            }
            QPushButton:hover { background-color: #45a049; }
            QRadioButton { color: white; font-size: 14px; margin: 5px; }
            QCheckBox { color: white; font-size: 14px; margin: 5px; }
            #imageLabel { background-color: #333; border: 2px dashed #666; min-height: 300px; }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)

        # Widgets
        self.label_instruction = QLabel("1. Select an image:")
        self.btn_choose = QPushButton("Choose Image")
        self.btn_choose.clicked.connect(self.choose_image)
        
        self.label_image = QLabel()
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        
        # Grayscale conversion option
        self.grayscale_checkbox = QCheckBox("Convert to grayscale before processing")
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        
        # Preview checkbox
        self.preview_checkbox = QCheckBox("Show preview (slower but shows intermediate results)")
        
        # Options layout
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.grayscale_checkbox)
        options_layout.addWidget(self.preview_checkbox)
        
        self.label_options = QLabel("2. Choose processing method:")
        self.option_quantize = QRadioButton("Median Quantization (32 colors)")
        self.option_bit_reduce = QRadioButton("Quantization + Bit Reduction (5-bit packed)")
        self.option_revert = QRadioButton("Revert Bit Reduction (5-bit → 8-bit median)")
        self.option_quantize.setChecked(True)
        
        # Compression info
        self.compression_label = QLabel("Estimated compression ratio: -")
        self.compression_label.setAlignment(Qt.AlignCenter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        self.btn_process = QPushButton("Process & Save")
        self.btn_process.clicked.connect(self.process_image)
        
        # Layout
        layout.addWidget(self.label_instruction)
        layout.addWidget(self.btn_choose)
        layout.addWidget(self.label_image)
        layout.addLayout(options_layout)
        layout.addWidget(self.label_options)
        layout.addWidget(self.option_quantize)
        layout.addWidget(self.option_bit_reduce)
        layout.addWidget(self.option_revert)
        layout.addWidget(self.compression_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.btn_process)

    def toggle_grayscale(self, state):
        self.convert_to_grayscale = state == Qt.Checked

    def choose_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.update_compression_estimate()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value >= 100:
            self.progress_bar.setVisible(False)

    def update_compression_estimate(self):
        if not self.image_path:
            return
        
        try:
            img = Image.open(self.image_path)
            original_size = img.size[0] * img.size[1] * (3 if img.mode == 'RGB' else 1)
            
            if self.option_bit_reduce.isChecked():
                # 5/8 of original size for each channel
                ratio = (5/8) * (1 if self.convert_to_grayscale else 3)
                estimated_size = original_size * ratio
                self.compression_label.setText(
                    f"Estimated compression ratio: {original_size/estimated_size:.2f}:1"
                )
            else:
                self.compression_label.setText("Estimated compression ratio: -")
        except:
            self.compression_label.setText("Could not estimate compression")

    def process_image(self):
        """Process image based on selected option."""
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first!")
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # Update UI
            
            img = Image.open(self.image_path)
            
            # Handle grayscale conversion if requested
            if self.convert_to_grayscale and img.mode != 'L':
                img = img.convert('L')
            
            img_array = np.array(img)
            self.is_color = img_array.ndim == 3 and not self.convert_to_grayscale
            
            start_time = time.time()
            
            if self.option_quantize.isChecked():
                self.process_quantization(img_array)
            elif self.option_bit_reduce.isChecked():
                self.process_bit_reduction(img_array)
            elif self.option_revert.isChecked():
                self.process_reversion()
            else:
                QMessageBox.warning(self, "Error", "Please select a processing method!")
                return
            
            # Save the image
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG (*.png);;JPEG (*.jpg);;Custom Format (*.iqf);;All Files (*)"
            )
            
            if save_path:
                if save_path.endswith('.iqf'):
                    self.save_custom_format(save_path)
                else:
                    self.processed_img.save(save_path)
                
                elapsed = time.time() - start_time
                QMessageBox.information(
                    self, "Success", 
                    f"Image saved to:\n{save_path}\nProcessing time: {elapsed:.2f} seconds"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    def process_quantization(self, img_array):
        """Process median quantization."""
        if self.is_color:
            quantized = np.stack([
                apply_median_quantization(img_array[:, :, 0], lambda v: self.update_progress(v/3)),
                apply_median_quantization(img_array[:, :, 1], lambda v: self.update_progress(33 + v/3)),
                apply_median_quantization(img_array[:, :, 2], lambda v: self.update_progress(66 + v/3))
            ], axis=-1)
        else:
            quantized = apply_median_quantization(img_array, self.update_progress)
        
        self.processed_img = Image.fromarray(quantized.astype(np.uint8))
        
        if self.preview_checkbox.isChecked():
            preview = Image.fromarray(quantized.astype(np.uint8))
            preview = preview.resize((400, 400), Image.LANCZOS)
            self.label_image.setPixmap(QPixmap.fromImage(preview.toqimage()))

    def process_bit_reduction(self, img_array):
        """Process bit reduction."""
        if self.is_color:
            # Process each channel in parallel would be ideal, but for simplicity we do sequentially
            quantized = np.stack([
                apply_median_quantization(img_array[:, :, 0]),
                apply_median_quantization(img_array[:, :, 1]),
                apply_median_quantization(img_array[:, :, 2])
            ], axis=-1)
            
            packed_r, shape_r, pad_r = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 0]))
            packed_g, shape_g, pad_g = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 1]))
            packed_b, shape_b, pad_b = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 2]))
            
            self.packed_data = (packed_r, packed_g, packed_b)
            self.original_shape = shape_r
            self.pad_length = (pad_r, pad_g, pad_b)
            
            # For display, unpack one channel
            unpacked = unpack_bytes_to_5bit(packed_r, shape_r, pad_r)
            restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
            self.processed_img = Image.fromarray(restored.astype(np.uint8))
        else:
            quantized = apply_median_quantization(img_array)
            five_bit = reduce_to_5bit(quantized)
            self.packed_data, self.original_shape, self.pad_length = pack_5bit_to_bytes(five_bit)
            
            # For display
            unpacked = unpack_bytes_to_5bit(self.packed_data, self.original_shape, self.pad_length)
            restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
            self.processed_img = Image.fromarray(restored.astype(np.uint8))
        
        self.update_progress(100)
        
        if self.preview_checkbox.isChecked():
            preview = self.processed_img.resize((400, 400), Image.LANCZOS)
            self.label_image.setPixmap(QPixmap.fromImage(preview.toqimage()))

    def process_reversion(self):
        """Process reversion from packed data."""
        if not hasattr(self, 'packed_data'):
            QMessageBox.warning(self, "Error", "No packed data to revert!")
            return
        
        if self.is_color:
            # Revert each channel for color images
            r_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                unpack_bytes_to_5bit(self.packed_data[0], self.original_shape, self.pad_length[0]))
            g_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                unpack_bytes_to_5bit(self.packed_data[1], self.original_shape, self.pad_length[1]))
            b_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                unpack_bytes_to_5bit(self.packed_data[2], self.original_shape, self.pad_length[2]))
            restored = np.stack([r_restored, g_restored, b_restored], axis=-1)
        else:
            restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                unpack_bytes_to_5bit(self.packed_data, self.original_shape, self.pad_length))
        
        self.processed_img = Image.fromarray(restored.astype(np.uint8))
        self.update_progress(100)
        
        if self.preview_checkbox.isChecked():
            preview = self.processed_img.resize((400, 400), Image.LANCZOS)
            self.label_image.setPixmap(QPixmap.fromImage(preview.toqimage()))

    def save_custom_format(self, file_path):
        """Save in custom IQF (Image Quantization Format)."""
        metadata = {
            'original_shape': self.original_shape,
            'is_color': self.is_color,
            'pad_length': self.pad_length if hasattr(self, 'pad_length') else 0,
            'version': '1.0'
        }
        
        with open(file_path, 'wb') as f:
            # Write metadata as JSON
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            f.write(len(metadata_bytes).to_bytes(4, 'big'))
            f.write(metadata_bytes)
            
            # Write packed data
            if self.is_color:
                for channel in self.packed_data:
                    f.write(len(channel).to_bytes(4, 'big'))
                    f.write(channel)
            else:
                f.write(len(self.packed_data).to_bytes(4, 'big'))
                f.write(self.packed_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())