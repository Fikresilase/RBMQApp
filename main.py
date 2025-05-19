import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

# Median values for quantization (32 groups)
MEDIAN_VALUES = [
    4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92,
    100, 108, 116, 124, 132, 140, 148, 156, 164, 172,
    180, 188, 196, 204, 212, 220, 228, 236, 244, 252
]

# Create mappings
MEDIAN_TO_INDEX = {median: idx for idx, median in enumerate(MEDIAN_VALUES)}
INDEX_TO_MEDIAN = {idx: median for idx, median in enumerate(MEDIAN_VALUES)}

def apply_median_quantization(img_array):
    """Quantize image to nearest median value (32 levels)."""
    quantized = np.zeros_like(img_array)
    for median in MEDIAN_VALUES:
        lower = median - 4
        upper = median + 3
        quantized[(img_array >= lower) & (img_array <= upper)] = median
    return quantized

def reduce_to_5bit(quantized_img):
    """Convert median values to 5-bit indices (0-31)."""
    return np.vectorize(MEDIAN_TO_INDEX.get)(quantized_img)

def pack_5bit_to_bytes(five_bit_img):
    """Pack 5-bit indices into tightly packed bytes."""
    flat = five_bit_img.flatten()
    packed = bytearray()
    buffer = 0
    bits_in_buffer = 0
    
    for value in flat:
        buffer = (buffer << 5) | value
        bits_in_buffer += 5
        while bits_in_buffer >= 8:
            packed.append((buffer >> (bits_in_buffer - 8)) & 0xFF)
            bits_in_buffer -= 8
            buffer &= (1 << bits_in_buffer) - 1
    
    if bits_in_buffer > 0:
        packed.append((buffer << (8 - bits_in_buffer)) & 0xFF)
    
    return packed, five_bit_img.shape

def unpack_bytes_to_5bit(packed_data, original_shape):
    """Unpack bytes back to 5-bit indices."""
    total_values = original_shape[0] * original_shape[1]
    unpacked = []
    buffer = 0
    bits_in_buffer = 0
    
    for byte in packed_data:
        buffer = (buffer << 8) | byte
        bits_in_buffer += 8
        while bits_in_buffer >= 5 and len(unpacked) < total_values:
            value = (buffer >> (bits_in_buffer - 5)) & 0b11111
            unpacked.append(value)
            bits_in_buffer -= 5
            buffer &= (1 << bits_in_buffer) - 1
    
    return np.array(unpacked, dtype=np.uint8).reshape(original_shape)

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.processed_img = None
        self.packed_data = None
        self.original_shape = None
        self.is_color = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Bitstream Processor")
        self.setGeometry(100, 100, 800, 600)
        
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
            #imageLabel { background-color: #333; border: 2px dashed #666; min-height: 300px; }
        """)

        # Widgets
        self.label_instruction = QLabel("1. Select an image:")
        self.btn_choose = QPushButton("Choose Image")
        self.btn_choose.clicked.connect(self.choose_image)
        
        self.label_image = QLabel()
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        
        self.label_options = QLabel("2. Choose processing method:")
        self.option_quantize = QRadioButton("Median Quantization (32 colors)")
        self.option_bit_reduce = QRadioButton("Quantization + Bit Reduction (5-bit packed)")
        self.option_revert = QRadioButton("Revert Bit Reduction (5-bit → 8-bit median)")
        
        self.btn_process = QPushButton("Process & Save")
        self.btn_process.clicked.connect(self.process_image)
        
        # Layout
        layout.addWidget(self.label_instruction)
        layout.addWidget(self.btn_choose)
        layout.addWidget(self.label_image)
        layout.addWidget(self.label_options)
        layout.addWidget(self.option_quantize)
        layout.addWidget(self.option_bit_reduce)
        layout.addWidget(self.option_revert)
        layout.addWidget(self.btn_process)

    def choose_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def process_image(self):
        """Process image based on selected option."""
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first!")
            return
        
        try:
            img = Image.open(self.image_path)
            img_array = np.array(img)
            self.is_color = img_array.ndim == 3  # Check if color image
            
            if self.option_quantize.isChecked():
                if self.is_color:
                    # Process each channel separately for color images
                    quantized = np.stack([
                        apply_median_quantization(img_array[:, :, 0]),
                        apply_median_quantization(img_array[:, :, 1]),
                        apply_median_quantization(img_array[:, :, 2])
                    ], axis=-1)
                else:
                    quantized = apply_median_quantization(img_array)
                self.processed_img = Image.fromarray(quantized.astype(np.uint8))
                
            elif self.option_bit_reduce.isChecked():
                if self.is_color:
                    # Quantize and pack each channel
                    quantized = np.stack([
                        apply_median_quantization(img_array[:, :, 0]),
                        apply_median_quantization(img_array[:, :, 1]),
                        apply_median_quantization(img_array[:, :, 2])
                    ], axis=-1)
                    packed_r, shape = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 0]))
                    packed_g, _ = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 1]))
                    packed_b, _ = pack_5bit_to_bytes(reduce_to_5bit(quantized[:, :, 2]))
                    self.packed_data = (packed_r, packed_g, packed_b)
                    self.original_shape = shape
                    
                    # For display, unpack one channel
                    unpacked = unpack_bytes_to_5bit(packed_r, shape)
                    restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
                    self.processed_img = Image.fromarray(restored.astype(np.uint8))
                else:
                    quantized = apply_median_quantization(img_array)
                    five_bit = reduce_to_5bit(quantized)
                    self.packed_data, self.original_shape = pack_5bit_to_bytes(five_bit)
                    
                    # For display
                    unpacked = unpack_bytes_to_5bit(self.packed_data, self.original_shape)
                    restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
                    self.processed_img = Image.fromarray(restored.astype(np.uint8))
                
            elif self.option_revert.isChecked():
                if not hasattr(self, 'packed_data'):
                    QMessageBox.warning(self, "Error", "No packed data to revert!")
                    return
                
                if self.is_color:
                    # Revert each channel for color images
                    r_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                        unpack_bytes_to_5bit(self.packed_data[0], self.original_shape))
                    g_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                        unpack_bytes_to_5bit(self.packed_data[1], self.original_shape))
                    b_restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                        unpack_bytes_to_5bit(self.packed_data[2], self.original_shape))
                    restored = np.stack([r_restored, g_restored, b_restored], axis=-1)
                else:
                    restored = np.vectorize(INDEX_TO_MEDIAN.get)(
                        unpack_bytes_to_5bit(self.packed_data, self.original_shape))
                self.processed_img = Image.fromarray(restored.astype(np.uint8))
                
            else:
                QMessageBox.warning(self, "Error", "Please select a processing method!")
                return
            
            # Save the image
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
            )
            if save_path:
                self.processed_img.save(save_path)
                QMessageBox.information(self, "Success", f"Image saved to:\n{save_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())