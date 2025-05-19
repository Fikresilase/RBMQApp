import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QCheckBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import time
import json

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
    # Pad with zeros if length not divisible by 5
    pad_length = (5 - (len(flat) % 5)) % 5  # Fixed parenthesis
    padded = np.pad(flat, (0, pad_length), mode='constant')  # Fixed np.pad call
    
    # Reshape to group 5 values (25 bits)
    grouped = padded.reshape(-1, 5)
    
    # Convert each group to bytes
    packed = bytearray()
    for group in grouped:
        # Combine 5 5-bit values into 25 bits
        combined = 0
        for i, val in enumerate(group):
            combined |= val << (20 - i*5)
        
        # Split into 3 bytes (24 bits) + 1 byte (remaining 1 bit)
        packed.append((combined >> 16) & 0xFF)
        packed.append((combined >> 8) & 0xFF)
        packed.append(combined & 0xFF)
        if pad_length > 0:
            packed.append((combined >> 24) & 0x01)
    
    return bytes(packed), five_bit_img.shape, pad_length

def unpack_bytes_to_5bit(packed_data, original_shape, pad_length):
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
        self.pad_length = 0
        self.is_color = False
        self.convert_to_grayscale = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Advanced Image Quantizer")
        self.setGeometry(100, 100, 800, 700)
        
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
        
        # Options layout
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.grayscale_checkbox)
        
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
        except Exception as e:
            print(f"Error estimating compression: {e}")
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
                "PNG (*.png);;JPEG (*.jpg);;Binary File (*.bin);;All Files (*)"
            )
            
            if save_path:
                if save_path.endswith('.bin'):
                    self.save_binary_file(save_path)
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
                apply_median_quantization(img_array[:, :, 0]),
                apply_median_quantization(img_array[:, :, 1]),
                apply_median_quantization(img_array[:, :, 2])
            ], axis=-1)
        else:
            quantized = apply_median_quantization(img_array)
        
        self.processed_img = Image.fromarray(quantized.astype(np.uint8))
        self.update_progress(100)

    def process_bit_reduction(self, img_array):
        """Process bit reduction with step-by-step display and binary file saving."""
        try:
            print("\n=== Starting Quantization + Bit Reduction Process ===")
            
            # 1. Apply Median Quantization
            print("\nStep 1: Applying Median Quantization...")
            if self.is_color:
                quantized = np.stack([
                    apply_median_quantization(img_array[:, :, 0]),
                    apply_median_quantization(img_array[:, :, 1]),
                    apply_median_quantization(img_array[:, :, 2])
                ], axis=-1)
                # For display, we'll just show the red channel
                channel_quantized = quantized[:, :, 0]
            else:
                quantized = apply_median_quantization(img_array)
                channel_quantized = quantized
            
            # Display first 10 pixel values and their binary
            print("\nQuantized values (first 10 pixels):")
            for i in range(min(10, channel_quantized.size)):
                val = channel_quantized.flat[i]
                print(f"Pixel {i}: {val} (binary: {val:08b})")
            
            # 2. Map to 5-bit indices (remove 3 LSBs)
            print("\nStep 2: Mapping to 5-bit indices (removing 3 LSBs)...")
            five_bit = reduce_to_5bit(quantized)
            if self.is_color:
                channel_five_bit = five_bit[:, :, 0]
            else:
                channel_five_bit = five_bit
            
            # Display first 10 mapped values and their binary
            print("\n5-bit mapped values (first 10 pixels):")
            for i in range(min(10, channel_five_bit.size)):
                val = channel_five_bit.flat[i]
                print(f"Pixel {i}: {val} (binary: {val:05b})")
            
            # 3. Pack into bitstream
            print("\nStep 3: Packing into bitstream...")
            if self.is_color:
                packed_r, shape_r, pad_r = pack_5bit_to_bytes(five_bit[:, :, 0])
                packed_g, shape_g, pad_g = pack_5bit_to_bytes(five_bit[:, :, 1])
                packed_b, shape_b, pad_b = pack_5bit_to_bytes(five_bit[:, :, 2])
                self.packed_data = (packed_r, packed_g, packed_b)
                self.original_shape = shape_r
                self.pad_length = (pad_r, pad_g, pad_b)
                packed = packed_r  # For display, use red channel
            else:
                self.packed_data, self.original_shape, self.pad_length = pack_5bit_to_bytes(five_bit)
                packed = self.packed_data
            
            # Display first 10 bytes of packed data
            print("\nPacked data (first 10 bytes):")
            for i in range(min(10, len(packed))):
                print(f"Byte {i}: {packed[i]:08b}")
            
            # 4. Save to .bin file automatically
            print("\nStep 4: Saving to binary file...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            bin_filename = f"bitreduced_{timestamp}.bin"
            
            with open(bin_filename, 'wb') as f:
                if self.is_color:
                    for channel in self.packed_data:
                        f.write(channel)
                else:
                    f.write(self.packed_data)
            
            print(f"Saved packed data to {bin_filename}")
            
            # For display, unpack one channel
            if self.is_color:
                unpacked = unpack_bytes_to_5bit(packed_r, shape_r, pad_r)
                restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
            else:
                unpacked = unpack_bytes_to_5bit(self.packed_data, self.original_shape, self.pad_length)
                restored = np.vectorize(INDEX_TO_MEDIAN.get)(unpacked)
            
            self.processed_img = Image.fromarray(restored.astype(np.uint8))
            self.update_progress(100)
            
            print("\n=== Process Completed Successfully ===")
            
        except Exception as e:
            print(f"\n!!! Error in bit reduction process: {str(e)}")
            raise

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

    def save_binary_file(self, file_path):
        """Save packed data to binary file."""
        with open(file_path, 'wb') as f:
            if self.is_color:
                for channel in self.packed_data:
                    f.write(channel)
            else:
                f.write(self.packed_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())