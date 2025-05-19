import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, 
    QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import struct

# Median values for quantization (32 groups, width of 8)
MEDIAN_VALUES = [
    4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 
    100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 
    180, 188, 196, 204, 212, 220, 228, 236, 244, 252
]

# Create a lookup table for 5-bit → median value
BIT_TO_MEDIAN = {i: median for i, median in enumerate(MEDIAN_VALUES)}

def apply_median_quantization(img_array):
    """Quantize image to nearest median value (32 levels)."""
    quantized = np.zeros_like(img_array)
    for i, median in enumerate(MEDIAN_VALUES):
        lower = i * 8
        upper = lower + 7
        quantized[(img_array >= lower) & (img_array <= upper)] = median
    return quantized

def pack_5bit_values(quantized_img):
    """
    Compress quantized image (8-bit) into tightly packed 5-bit values.
    Returns a byte array.
    """
    # Convert median values to 5-bit indices (0-31)
    median_to_bit = {median: i for i, median in enumerate(MEDIAN_VALUES)}
    bit_indices = np.vectorize(median_to_bit.get)(quantized_img)
    
    # Flatten and pack into bytes (5 bits per value)
    flat_bits = bit_indices.flatten()
    packed_data = bytearray()
    buffer = 0
    bits_in_buffer = 0
    
    for value in flat_bits:
        buffer = (buffer << 5) | value
        bits_in_buffer += 5
        if bits_in_buffer >= 8:
            packed_data.append((buffer >> (bits_in_buffer - 8)) & 0xFF)
            buffer &= (1 << (bits_in_buffer - 8)) - 1
            bits_in_buffer -= 8
    
    # Handle remaining bits
    if bits_in_buffer > 0:
        packed_data.append((buffer << (8 - bits_in_buffer)) & 0xFF)
    
    return packed_data, quantized_img.shape

def unpack_5bit_values(packed_data, original_shape):
    """
    Unpack tightly packed 5-bit values back into 8-bit median values.
    """
    total_values = original_shape[0] * original_shape[1]
    unpacked_bits = []
    buffer = 0
    bits_in_buffer = 0
    
    for byte in packed_data:
        buffer = (buffer << 8) | byte
        bits_in_buffer += 8
        while bits_in_buffer >= 5 and len(unpacked_bits) < total_values:
            value = (buffer >> (bits_in_buffer - 5)) & 0b11111
            unpacked_bits.append(value)
            bits_in_buffer -= 5
            buffer &= (1 << bits_in_buffer) - 1
    
    # Map 5-bit values back to median
    restored_img = np.array([BIT_TO_MEDIAN[v] for v in unpacked_bits], dtype=np.uint8)
    return restored_img.reshape(original_shape)

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.processed_img = None
        self.packed_data = None
        self.original_shape = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Advanced Image Quantizer")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { 
                font-size: 16px; 
                color: #ffffff; 
                margin: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #45a049; }
            QRadioButton { 
                font-size: 14px; 
                color: #ffffff; 
                margin: 5px;
            }
            #imageLabel { 
                background-color: #333; 
                border: 2px dashed #666; 
                min-height: 300px;
            }
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
        
        self.btn_process = QPushButton("Process & Save Image")
        self.btn_process.clicked.connect(self.process_image)
        
        # Add widgets to layout
        layout.addWidget(self.label_instruction)
        layout.addWidget(self.btn_choose)
        layout.addWidget(self.label_image)
        layout.addWidget(self.label_options)
        layout.addWidget(self.option_quantize)
        layout.addWidget(self.option_bit_reduce)
        layout.addWidget(self.option_revert)
        layout.addWidget(self.btn_process)

    def choose_image(self):
        """Open file dialog to select an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.label_image.setPixmap(
                pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def process_image(self):
        """Process the image based on selected option."""
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first!")
            return
        
        try:
            img = Image.open(self.image_path)
            img_array = np.array(img)
            
            if img_array.ndim == 2:  # Convert grayscale to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            if self.option_quantize.isChecked():
                r_processed = apply_median_quantization(r)
                g_processed = apply_median_quantization(g)
                b_processed = apply_median_quantization(b)
                processed_img = np.stack([r_processed, g_processed, b_processed], axis=-1)
                self.processed_img = Image.fromarray(processed_img.astype(np.uint8))
                
            elif self.option_bit_reduce.isChecked():
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                
                # Pack each channel into 5-bit binary
                r_packed, r_shape = pack_5bit_values(r_quant)
                g_packed, g_shape = pack_5bit_values(g_quant)
                b_packed, b_shape = pack_5bit_values(b_quant)
                
                # Save packed data (for demonstration, we'll just store it)
                self.packed_data = (r_packed, g_packed, b_packed)
                self.original_shape = r_shape  # Assuming all channels have same shape
                
                # For visualization, we can unpack it again (just to show in UI)
                r_unpacked = unpack_5bit_values(r_packed, r_shape)
                g_unpacked = unpack_5bit_values(g_packed, g_shape)
                b_unpacked = unpack_5bit_values(b_packed, b_shape)
                
                processed_img = np.stack([r_unpacked, g_unpacked, b_unpacked], axis=-1)
                self.processed_img = Image.fromarray(processed_img.astype(np.uint8))
                
            elif self.option_revert.isChecked():
                if not hasattr(self, 'packed_data'):
                    QMessageBox.warning(self, "Error", "No packed data to revert!")
                    return
                
                r_packed, g_packed, b_packed = self.packed_data
                r_restored = unpack_5bit_values(r_packed, self.original_shape)
                g_restored = unpack_5bit_values(g_packed, self.original_shape)
                b_restored = unpack_5bit_values(b_packed, self.original_shape)
                
                processed_img = np.stack([r_restored, g_restored, b_restored], axis=-1)
                self.processed_img = Image.fromarray(processed_img.astype(np.uint8))
                
            else:
                QMessageBox.warning(self, "Error", "Please select a processing method!")
                return
            
            # Save the image
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
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