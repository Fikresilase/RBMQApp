import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QVBoxLayout, QWidget, QHBoxLayout, QStackedWidget,
                            QMessageBox, QRadioButton)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
from PIL import Image

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

class CompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
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
        
        self.label_options = QLabel('2. Choose an option')
        self.label_options.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_options)
        
        self.option1 = QRadioButton('Apply Median Quantization')
        self.option2 = QRadioButton('Apply Quantization + Bit Reduction')
        layout.addWidget(self.option1, alignment=Qt.AlignCenter)
        layout.addWidget(self.option2, alignment=Qt.AlignCenter)
        
        self.btn_process = QPushButton('Process and Save Image')
        self.btn_process.clicked.connect(self.process_and_save_image)
        layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)
    
    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 
                                                        'Image Files (*.png *.jpg *.jpeg);;All Files (*)', 
                                                        options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def process_and_save_image(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image first.')
            return
        
        option = 1 if self.option1.isChecked() else 2 if self.option2.isChecked()  else None
        if option is None:
            QMessageBox.warning(self, 'Error', 'Please choose an option.')
            return
        
        try:
            img = Image.open(self.image_path)
            img_array = np.array(img)  # Get RGB values
            
            if img_array.ndim == 2:  # If grayscale, convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            if option == 1:
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                processed_img_array = np.stack([r_quant, g_quant, b_quant], axis=-1)
            elif option == 2:
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                r_reduced = apply_bit_reduction(r_quant)
                g_reduced = apply_bit_reduction(g_quant)
                b_reduced = apply_bit_reduction(b_quant)
                processed_img_array = np.stack([r_reduced, g_reduced, b_reduced], axis=-1)
            elif option == 3:
                r_reduced = apply_bit_reduction(r)
                g_reduced = apply_bit_reduction(g)
                b_reduced = apply_bit_reduction(b)
                r_reverted = revert_bit_reduction(r_reduced)
                g_reverted = revert_bit_reduction(g_reduced)
                b_reverted = revert_bit_reduction(b_reduced)
                processed_img_array = np.stack([r_reverted, g_reverted, b_reverted], axis=-1)
            
            processed_img = Image.fromarray(processed_img_array.astype(np.uint8))
            
            save_as, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 
                                                   'PNG files (*.png);;JPEG files (*.jpeg);;All Files (*)')
            if save_as:
                processed_img.save(save_as)
                QMessageBox.information(self, 'Success', f'Image saved as {save_as}')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")

class DecompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_load = QPushButton('Load Image')
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_process = QPushButton('Process Bits')
        self.btn_process.clicked.connect(self.process_bits)
        self.btn_process.setEnabled(False)
        
        self.btn_save = QPushButton('Save Processed Image')
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_process)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)', 
            options=options)
            
        if file_path:
            self.image_path = file_path
            try:
                # Load with PIL first to handle all formats
                pil_img = Image.open(file_path)
                self.original_image = np.array(pil_img)
                
                # Convert to QPixmap for display
                if pil_img.mode == 'RGB':
                    bytes_per_line = 3 * pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_RGB888)
                elif pil_img.mode == 'RGBA':
                    bytes_per_line = 4 * pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_RGBA8888)
                else:  # Grayscale
                    bytes_per_line = pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_Grayscale8)
                
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.btn_process.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Error loading image: {e}")
    
    def process_bits(self):
        if self.original_image is None:
            return
            
        try:
            # Process bits: shift left 3 and add 4
            self.processed_image = (self.original_image << 3) + 4
            
            # Clip values to 0-255 range
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            
            # Display the processed image
            if len(self.processed_image.shape) == 2:  # Grayscale
                bytes_per_line = self.processed_image.shape[1]
                qimage = QImage(self.processed_image.data, self.processed_image.shape[1], 
                              self.processed_image.shape[0], bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color
                bytes_per_line = 3 * self.processed_image.shape[1]
                qimage = QImage(self.processed_image.data, self.processed_image.shape[1], 
                              self.processed_image.shape[0], bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_save.setEnabled(True)
            QMessageBox.information(self, 'Success', 'Processing complete!')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error processing image: {e}")
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)', 
            options=options)
            
        if file_path:
            try:
                # Convert numpy array to PIL Image
                if len(self.processed_image.shape) == 2:  # Grayscale
                    processed_img = Image.fromarray(self.processed_image, mode='L')
                else:  # Color
                    processed_img = Image.fromarray(self.processed_image, mode='RGB')
                
                # Save with appropriate format based on extension
                processed_img.save(file_path)
                QMessageBox.information(self, 'Success', f'Image saved to: {file_path}')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Error saving image: {e}")

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
        self.compress_btn = QPushButton('Compress an Image')
        self.compress_btn.setCheckable(True)
        self.compress_btn.setChecked(True)
        
        self.decompress_btn = QPushButton('Decompress an Image')
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