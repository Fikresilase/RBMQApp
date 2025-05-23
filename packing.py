import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QFileDialog, QLabel, QRadioButton, 
                             QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from scipy.io import savemat
import zlib

# Median values for quantization (32 levels)
median_values = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 
                100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 
                180, 188, 196, 204, 212, 220, 228, 236, 244, 252]

def print_pixel_samples(title, array):
    """Print sample 5x5 block of pixel values"""
    print(f"\n{title}:")
    print(array[:5, :5])

def apply_median_quantization(img_array):
    """Quantize to 32 median values (5-bit)"""
    quantized = np.zeros_like(img_array)
    for i, median in enumerate(median_values):
        lower = i * 8
        upper = lower + 7
        quantized[(img_array >= lower) & (img_array <= upper)] = median
    return quantized

def reduce_to_5bit(img_array):
    """Convert 8-bit to 5-bit values (0-31)"""
    return np.right_shift(img_array, 3)

def pack_5bit_array(data):
    """Pack 5-bit values into bytes (8 values -> 5 bytes)"""
    flat = data.flatten()
    extra = len(flat) % 8
    if extra:
        flat = np.pad(flat, (0, 8 - extra), 'constant')
    
    packed = np.zeros((len(flat) * 5 // 8,), dtype=np.uint8)
    
    for i in range(len(flat) // 8):
        vals = flat[i*8:(i+1)*8]
        packed[i*5] = (vals[0] << 3) | (vals[1] >> 2)
        packed[i*5+1] = ((vals[1] & 0x03) << 6) | (vals[2] << 1) | (vals[3] >> 4)
        packed[i*5+2] = ((vals[3] & 0x0F) << 4) | (vals[4] >> 1)
        packed[i*5+3] = ((vals[4] & 0x01) << 7) | (vals[5] << 2) | (vals[6] >> 3)
        packed[i*5+4] = ((vals[6] & 0x07) << 5) | vals[7]
    
    return packed, data.shape

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.original_shape = None

    def initUI(self):
        self.setWindowTitle('5-bit Image Compressor')
        self.setGeometry(100, 100, 600, 500)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # UI Elements
        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("border: 2px dashed #aaa;")
        
        btn_load = QPushButton('Load Image')
        btn_load.clicked.connect(self.load_image)
        
        self.option_quant = QRadioButton('Median Quantization (32 levels)')
        self.option_bitred = QRadioButton('5-bit Reduction')
        self.option_bitred.setChecked(True)
        
        btn_process = QPushButton('Compress and Save')
        btn_process.clicked.connect(self.compress_and_save)
        
        # Layout
        layout.addWidget(QLabel('Image Preview:'))
        layout.addWidget(self.label_img)
        layout.addWidget(btn_load)
        layout.addWidget(QLabel('Compression Method:'))
        layout.addWidget(self.option_quant)
        layout.addWidget(self.option_bitred)
        layout.addWidget(btn_process)
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 
            'Images (*.png *.jpg *.jpeg);;All Files (*)')
        
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.label_img.setPixmap(
                pixmap.scaled(400, 300, Qt.KeepAspectRatio))
            
            # Print original pixel samples
            img = np.array(Image.open(path))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            print("\nOriginal Image Samples (RGB):")
            print_pixel_samples("Red", img[:,:,0])
            print_pixel_samples("Green", img[:,:,1])
            print_pixel_samples("Blue", img[:,:,2])

    def compress_and_save(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'No image loaded!')
            return
        
        try:
            img = np.array(Image.open(self.image_path))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            
            # Process based on selected option
            if self.option_quant.isChecked():
                processed = np.stack([
                    apply_median_quantization(img[:,:,0]),
                    apply_median_quantization(img[:,:,1]),
                    apply_median_quantization(img[:,:,2])
                ], axis=-1)
                print("\nAfter Median Quantization:")
            else:
                processed = np.stack([
                    reduce_to_5bit(img[:,:,0]),
                    reduce_to_5bit(img[:,:,1]),
                    reduce_to_5bit(img[:,:,2])
                ], axis=-1)
                print("\nAfter 5-bit Reduction:")
            
            # Print processed samples
            print_pixel_samples("Red", processed[:,:,0])
            print_pixel_samples("Green", processed[:,:,1])
            print_pixel_samples("Blue", processed[:,:,2])
            
            # Pack 5-bit data efficiently
            packed, shape = pack_5bit_array(processed)
            compressed = zlib.compress(packed.tobytes())
            
            # Save options
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save Compressed Data', '', 
                'MAT Files (*.mat);;All Files (*)')
            
            if path:
                if not path.endswith('.mat'):
                    path += '.mat'
                
                savemat(path, {
                    'compressed_data': np.frombuffer(compressed, dtype=np.uint8),
                    'original_shape': shape,
                    'is_quantized': self.option_quant.isChecked()
                }, do_compression=True)
                
                # Calculate stats
                orig_size = img.nbytes
                comp_size = len(compressed)
                ratio = orig_size / comp_size
                
                print(f"\nCompression Results:")
                print(f"Original: {orig_size/1024:.1f} KB")
                print(f"Compressed: {comp_size/1024:.1f} KB")
                print(f"Ratio: {ratio:.1f}x")
                
                QMessageBox.information(
                    self, 'Success', 
                    f'Saved 5-bit compressed data\n'
                    f'Compression ratio: {ratio:.1f}x')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            print("Error:", e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())