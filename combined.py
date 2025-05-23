import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QVBoxLayout, QWidget, QHBoxLayout, QStackedWidget,
                            QMessageBox, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from scipy.io import savemat, loadmat
import zlib

# Define the quantization median values for each group (32 groups with a width of 8)
median_values = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124,
                 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]

def apply_median_quantization(img_array):
    """Apply median quantization to the image.""" 
    quantized_array = np.zeros_like(img_array)
    for i, median in enumerate(median_values):
        lower_bound = i * 8
        upper_bound = lower_bound + 7
        quantized_array[(img_array >= lower_bound) & (img_array <= upper_bound)] = median
    return quantized_array

def apply_bit_reduction(img_array):
    """Reduce bit depth from 8 bits to 5 bits (32 levels)."""
    reduced_bit_image = np.right_shift(img_array, 3)  # Bit reduction
    return reduced_bit_image

def revert_bit_reduction(reduced_bit_image):
    """Revert from bit reduction (5 bits) to original quantized image."""
    reverted_image = np.left_shift(reduced_bit_image, 3)+4  # Reverting bit depth
    return reverted_image

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

class CompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Image selection
        self.label_instruction = QLabel('1. Select an image')
        self.label_instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_instruction)
        
        self.btn_choose = QPushButton('Choose Image')
        self.btn_choose.clicked.connect(self.choose_image)
        layout.addWidget(self.btn_choose, alignment=Qt.AlignCenter)
        
        self.label_image = QLabel()
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(400, 300)
        layout.addWidget(self.label_image, alignment=Qt.AlignCenter)
        
        # Compression options
        self.label_options = QLabel('2. Choose compression method')
        self.label_options.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_options)
        
        self.option1 = QRadioButton('Median Quantization')
        self.option2 = QRadioButton('Quantization + Bit Reduction')
        self.option1.setChecked(True)
        
        options_group = QVBoxLayout()
        options_group.addWidget(self.option1)
        options_group.addWidget(self.option2)
        layout.addLayout(options_group)
        
        # Save options
        self.label_save = QLabel('3. Choose output format')
        self.label_save.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_save)
        
        self.save_mat = QRadioButton('MAT File (compressed)')
        self.save_img = QRadioButton('Standard Image (PNG/JPEG)')
        self.save_mat.setChecked(True)
        
        save_group = QVBoxLayout()
        save_group.addWidget(self.save_mat)
        save_group.addWidget(self.save_img)
        layout.addLayout(save_group)
        
        # Process button
        self.btn_process = QPushButton('Process and Save')
        self.btn_process.clicked.connect(self.process_and_save)
        layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)
    
    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg);;All Files (*)', 
            options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def process_and_save(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image first.')
            return
        
        try:
            img = Image.open(self.image_path)
            img_array = np.array(img)
            
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Process image based on selected method
            if self.option1.isChecked():
                r = apply_median_quantization(img_array[:, :, 0])
                g = apply_median_quantization(img_array[:, :, 1])
                b = apply_median_quantization(img_array[:, :, 2])
                processed_img = np.stack([r, g, b], axis=-1)
            else:
                r = apply_bit_reduction(apply_median_quantization(img_array[:, :, 0]))
                g = apply_bit_reduction(apply_median_quantization(img_array[:, :, 1]))
                b = apply_bit_reduction(apply_median_quantization(img_array[:, :, 2]))
                processed_img = np.stack([r, g, b], axis=-1)
            
            # Save based on selected format
            if self.save_mat.isChecked():
                # Save as MAT file
                packed_data, original_shape = pack_5bit_array(processed_img)
                compressed_data = zlib.compress(packed_data.tobytes())
                
                save_path, _ = QFileDialog.getSaveFileName(
                    self, 'Save MAT File', '', 
                    'MAT Files (*.mat);;All Files (*)')
                
                if save_path:
                    if not save_path.endswith('.mat'):
                        save_path += '.mat'
                    
                    savemat(save_path, {
                        'compressed_data': np.frombuffer(compressed_data, dtype=np.uint8),
                        'original_shape': original_shape,
                        'is_quantized': self.option1.isChecked()
                    }, do_compression=True)
                    
                    QMessageBox.information(self, 'Success', 'MAT file saved successfully!')
            else:
                # Save as standard image
                save_path, _ = QFileDialog.getSaveFileName(
                    self, 'Save Image', '', 
                    'PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)')
                
                if save_path:
                    if self.option1.isChecked():
                        # For quantized images, values are already 8-bit
                        img_to_save = Image.fromarray(processed_img.astype(np.uint8))
                    else:
                        # For 5-bit reduced, convert back to 8-bit
                        img_to_save = Image.fromarray(revert_bit_reduction(processed_img))
                    
                    img_to_save.save(save_path)
                    QMessageBox.information(self, 'Success', 'Image saved successfully!')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")

class DecompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.reconstructed_img = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Input selection
        self.label_input = QLabel('1. Select input type')
        self.label_input.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_input)
        
        self.input_mat = QRadioButton('Load MAT File')
        self.input_img = QRadioButton('Load Standard Image')
        self.input_mat.setChecked(True)
        
        input_group = QVBoxLayout()
        input_group.addWidget(self.input_mat)
        input_group.addWidget(self.input_img)
        layout.addLayout(input_group)
        
        # Load button
        self.btn_load = QPushButton('Load File')
        self.btn_load.clicked.connect(self.load_file)
        layout.addWidget(self.btn_load, alignment=Qt.AlignCenter)
        
        # Image display
        self.label_image = QLabel()
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(400, 300)
        layout.addWidget(self.label_image, alignment=Qt.AlignCenter)
        
        # Process button
        self.btn_process = QPushButton('Process Image')
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
        
        # Save button
        self.btn_save = QPushButton('Save Image')
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)
    
    def load_file(self):
        options = QFileDialog.Options()
        
        if self.input_mat.isChecked():
            # Load MAT file
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Load MAT File', '', 
                'MAT Files (*.mat);;All Files (*)', 
                options=options)
            
            if file_path:
                try:
                    self.input_data = loadmat(file_path)
                    self.btn_process.setEnabled(True)
                    self.btn_save.setEnabled(False)
                    QMessageBox.information(self, 'Success', 'MAT file loaded successfully!')
                except Exception as e:
                    QMessageBox.critical(self, 'Error', f"Failed to load MAT file: {e}")
        else:
            # Load standard image
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Load Image', '', 
                'Image Files (*.png *.jpg *.jpeg);;All Files (*)', 
                options=options)
            
            if file_path:
                try:
                    img = Image.open(file_path)
                    self.input_data = np.array(img)
                    
                    if len(self.input_data.shape) == 2:
                        self.input_data = np.stack([self.input_data]*3, axis=-1)
                    
                    # Display the image
                    height, width, _ = self.input_data.shape
                    bytes_per_line = 3 * width
                    qimg = QImage(
                        self.input_data.data, 
                        width, height, bytes_per_line, 
                        QImage.Format_RGB888
                    )
                    self.label_image.setPixmap(QPixmap.fromImage(qimg))
                    self.btn_process.setEnabled(True)
                    self.btn_save.setEnabled(False)
                    
                    QMessageBox.information(self, 'Success', 'Image loaded successfully!')
                except Exception as e:
                    QMessageBox.critical(self, 'Error', f"Failed to load image: {e}")
    
    def process_image(self):
        if self.input_data is None:
            return
            
        try:
            if self.input_mat.isChecked():
                # Process MAT file
                compressed_data = self.input_data['compressed_data'].tobytes()
                decompressed = zlib.decompress(compressed_data)
                packed_data = np.frombuffer(decompressed, dtype=np.uint8)
                original_shape = self.input_data['original_shape'][0]
                unpacked_data = unpack_5bit_bytes(packed_data, original_shape)
                
                if self.input_data['is_quantized'][0][0]:
                    self.reconstructed_img = unpacked_data.astype(np.uint8)
                else:
                    self.reconstructed_img = revert_bit_reduction(unpacked_data)
            else:
                # Process standard image
                if np.max(self.input_data) <= 31:  # Likely 5-bit image
                    self.reconstructed_img = revert_bit_reduction(self.input_data)
                else:
                    self.reconstructed_img = self.input_data
            
            # Display the processed image
            height, width, _ = self.reconstructed_img.shape
            bytes_per_line = 3 * width
            qimg = QImage(
                self.reconstructed_img.data, 
                width, height, bytes_per_line, 
                QImage.Format_RGB888
            )
            self.label_image.setPixmap(QPixmap.fromImage(qimg))
            self.btn_save.setEnabled(True)
            
            QMessageBox.information(self, 'Success', 'Image processed successfully!')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Processing failed: {e}")
    
    def save_image(self):
        if self.reconstructed_img is None:
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)', 
            options=options)
        
        if file_path:
            try:
                Image.fromarray(self.reconstructed_img).save(file_path)
                QMessageBox.information(self, 'Success', f'Image saved to {file_path}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Failed to save image: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RBMQ Image Processor')
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(20)
        
        # Logo
        logo = QLabel('RBMQ')
        logo.setFont(QFont('Arial', 20, QFont.Bold))
        logo.setStyleSheet("color: #3498db;")
        sidebar_layout.addWidget(logo)
        
        # Sidebar buttons
        self.compress_btn = QPushButton('Compress')
        self.compress_btn.setCheckable(True)
        self.compress_btn.setChecked(True)
        
        self.decompress_btn = QPushButton('Decompress')
        self.decompress_btn.setCheckable(True)
        
        button_group = QVBoxLayout()
        button_group.addWidget(self.compress_btn)
        button_group.addWidget(self.decompress_btn)
        button_group.addStretch()
        
        sidebar_layout.addLayout(button_group)
        sidebar.setLayout(sidebar_layout)
        
        # Content area
        self.content = QStackedWidget()
        
        # Add pages
        self.compress_page = CompressPage()
        self.decompress_page = DecompressPage()
        
        self.content.addWidget(self.compress_page)
        self.content.addWidget(self.decompress_page)
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.content)
        
        main_widget.setLayout(main_layout)
        
        # Connect signals
        self.compress_btn.clicked.connect(self.show_compress)
        self.decompress_btn.clicked.connect(self.show_decompress)
        
        # Apply styles
        self.apply_styles()
    
    def show_compress(self):
        self.content.setCurrentIndex(0)
        self.compress_btn.setChecked(True)
        self.decompress_btn.setChecked(False)
    
    def show_decompress(self):
        self.content.setCurrentIndex(1)
        self.decompress_btn.setChecked(True)
        self.compress_btn.setChecked(False)
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a192f;
            }
            QWidget {
                background-color: #0a192f;
                color: #ccd6f6;
            }
            QPushButton {
                background-color: #1e3a8a;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                border: none;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #1e40af;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                font-weight: bold;
            }
            QLabel#imageLabel {
                background-color: #112240;
                border: 2px dashed #1e3a8a;
            }
            QRadioButton {
                font-size: 14px;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QMessageBox {
                background-color: #0a192f;
            }
            QMessageBox QLabel {
                color: #ccd6f6;
            }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())