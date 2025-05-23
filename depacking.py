import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QFileDialog, QLabel, QVBoxLayout, QWidget, 
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from scipy.io import loadmat
import zlib
from PIL import Image

def unpack_5bit_bytes(packed_data, original_shape):
    """Unpack bit-packed data back to 5-bit values"""
    total_values = original_shape[0] * original_shape[1] * original_shape[2]
    unpacked = np.zeros(total_values, dtype=np.uint8)
    
    # Calculate needed bytes (5 bytes per 8 values)
    needed_bytes = (total_values * 5 + 7) // 8
    packed_data = packed_data[:needed_bytes]
    
    for i in range(len(packed_data) // 5):
        byte0 = packed_data[i*5]
        byte1 = packed_data[i*5+1]
        byte2 = packed_data[i*5+2]
        byte3 = packed_data[i*5+3]
        byte4 = packed_data[i*5+4]
        
        # Unpack 8 values from 5 bytes
        unpacked[i*8] = byte0 >> 3
        unpacked[i*8+1] = ((byte0 & 0x07) << 2) | (byte1 >> 6)
        unpacked[i*8+2] = (byte1 >> 1) & 0x1F
        unpacked[i*8+3] = ((byte1 & 0x01) << 4) | (byte2 >> 4)
        unpacked[i*8+4] = ((byte2 & 0x0F) << 1) | (byte3 >> 7)
        unpacked[i*8+5] = (byte3 >> 2) & 0x1F
        unpacked[i*8+6] = ((byte3 & 0x03) << 3) | (byte4 >> 5)
        unpacked[i*8+7] = byte4 & 0x1F
    
    return unpacked[:total_values].reshape(original_shape)

class DecompressorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.mat_data = None
        self.reconstructed_img = None

    def initUI(self):
        self.setWindowTitle('5-bit Image Decompressor')
        self.setGeometry(100, 100, 600, 500)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # UI Elements
        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("border: 2px dashed #aaa;")
        
        btn_load = QPushButton('Load .mat File')
        btn_load.clicked.connect(self.load_mat_file)
        
        btn_decompress = QPushButton('Decompress and Show')
        btn_decompress.clicked.connect(self.decompress_and_show)
        
        btn_save = QPushButton('Save Reconstructed Image')
        btn_save.clicked.connect(self.save_reconstructed_image)
        
        # Info labels
        self.label_info = QLabel('No file loaded')
        self.label_info.setAlignment(Qt.AlignCenter)
        
        # Layout
        layout.addWidget(self.label_info)
        layout.addWidget(self.label_img)
        layout.addWidget(btn_load)
        layout.addWidget(btn_decompress)
        layout.addWidget(btn_save)
        
    def load_mat_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open MAT File', '', 
            'MAT Files (*.mat);;All Files (*)')
        
        if path:
            try:
                self.mat_data = loadmat(path)
                self.label_info.setText(f"Loaded: {path.split('/')[-1]}")
                
                # Print basic info
                print("\nLoaded MAT File Info:")
                print(f"Original shape: {self.mat_data['original_shape']}")
                print(f"Compression method: {'Quantized' if self.mat_data['is_quantized'] else '5-bit Reduced'}")
                print(f"Compressed size: {len(self.mat_data['compressed_data'])/1024:.1f} KB")
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Failed to load MAT file:\n{str(e)}")
    
    def decompress_and_show(self):
        if self.mat_data is None:
            QMessageBox.warning(self, 'Error', 'No MAT file loaded!')
            return
        
        try:
            # Decompress the data
            compressed_data = self.mat_data['compressed_data'].tobytes()
            decompressed = zlib.decompress(compressed_data)
            
            # Unpack the 5-bit data
            packed_data = np.frombuffer(decompressed, dtype=np.uint8)
            reconstructed = unpack_5bit_bytes(packed_data, 
                                           self.mat_data['original_shape'][0])
            
            # Convert back to 8-bit based on compression method
            if self.mat_data['is_quantized']:
                # For quantized images, values are already the median values
                self.reconstructed_img = reconstructed.astype(np.uint8)
            else:
                # For 5-bit reduced, shift back to 8-bit range
                self.reconstructed_img = np.left_shift(reconstructed, 3) + 4
            
            # Convert to QImage and display
            height, width, _ = self.reconstructed_img.shape
            bytes_per_line = 3 * width
            qimg = QImage(
                self.reconstructed_img.data, 
                width, height, bytes_per_line, 
                QImage.Format_RGB888
            )
            self.label_img.setPixmap(QPixmap.fromImage(qimg))
            
            # Print reconstruction info
            print("\nReconstruction Info:")
            print(f"Image shape: {self.reconstructed_img.shape}")
            print("Sample pixel values (top-left 5x5):")
            print(self.reconstructed_img[:5, :5, :])
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Decompression failed:\n{str(e)}")
    
    def save_reconstructed_image(self):
        if self.reconstructed_img is None:
            QMessageBox.warning(self, 'Error', 'No image to save!')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)')
        
        if path:
            try:
                Image.fromarray(self.reconstructed_img).save(path)
                QMessageBox.information(self, 'Success', f"Image saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Failed to save image:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DecompressorApp()
    window.show()
    sys.exit(app.exec_())