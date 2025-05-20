import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QVBoxLayout, QWidget, QTextEdit, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

class BitViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Bit Viewer & Processor')
        self.setGeometry(100, 100, 800, 800)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        self.layout.addWidget(self.image_label)
        
        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.layout.addWidget(self.console)
        
        # Buttons
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        
        self.process_button = QPushButton('Process Bits (Left Shift 3)')
        self.process_button.clicked.connect(self.process_bits)
        self.process_button.setEnabled(False)
        
        self.save_button = QPushButton('Save Processed Image')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.save_button)
        self.layout.addLayout(button_layout)
        
        # Image data storage
        self.original_image = None
        self.processed_image = None
        self.bit_values = None
        self.shifted_values = None

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)', options=options)
        
        if file_path:
            try:
                # Load image using PIL (handles all formats)
                img = Image.open(file_path)
                self.original_image = np.array(img)
                
                # Display image
                pixmap = QPixmap(file_path)
                self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))
                
                # Get and display bit values (first 10x10 pixels for demo)
                height, width = self.original_image.shape[:2]
                sample_size = min(10, height, width)
                
                # For RGB images, take first channel; for grayscale use as-is
                if len(self.original_image.shape) == 3:
                    sample_data = self.original_image[:sample_size, :sample_size, 0]
                else:
                    sample_data = self.original_image[:sample_size, :sample_size]
                
                self.bit_values = sample_data
                self.display_bit_values(sample_data, "Original Bit Values (Sample):")
                
                self.process_button.setEnabled(True)
                self.processed_image = None
                self.save_button.setEnabled(False)
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load image: {str(e)}')

    def display_bit_values(self, data, title):
        """Display bit values in the console"""
        self.console.append(f"\n{title}\n" + "="*40)
        
        # For 2D array (grayscale or single channel)
        if len(data.shape) == 2:
            for row in data:
                row_str = ' '.join([f'{pixel:03d}' for pixel in row])
                self.console.append(row_str)
        # For RGB (we're just showing first channel)
        else:
            for row in data:
                channel_str = ' '.join([f'{pixel[0]:03d}' for pixel in row])
                self.console.append(channel_str)

    def process_bits(self):
        if self.original_image is None:
            return
            
        try:
            # Process the image (left shift 3 bits)
            if len(self.original_image.shape) == 3:  # RGB
                self.processed_image = np.left_shift(self.original_image[:, :, :3], 3)
                sample_data = self.processed_image[:10, :10, 0]  # Sample first channel
            else:  # Grayscale
                self.processed_image = np.left_shift(self.original_image, 3)
                sample_data = self.processed_image[:10, :10]
            
            # Clip values to 0-255 range
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            
            # Display shifted values
            self.shifted_values = sample_data
            self.display_bit_values(sample_data, "Shifted Bit Values (Sample):")
            
            # Show processed image preview
            if len(self.processed_image.shape) == 3:
                img = Image.fromarray(self.processed_image)
            else:
                img = Image.fromarray(self.processed_image).convert('L')
                
            img.save('temp_processed.png')  # Temporary save for preview
            self.image_label.setPixmap(QPixmap('temp_processed.png').scaled(600, 400, Qt.KeepAspectRatio))
            
            self.save_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Processing failed: {str(e)}')

    def save_image(self):
        if self.processed_image is None:
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Processed Image', '', 
            'PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp);;TIFF Image (*.tiff)', 
            options=options)
        
        if file_path:
            try:
                if len(self.processed_image.shape) == 3:
                    img = Image.fromarray(self.processed_image)
                else:
                    img = Image.fromarray(self.processed_image).convert('L')
                
                # Determine format from extension
                if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                    img.save(file_path, 'JPEG', quality=95)
                else:
                    img.save(file_path)
                
                QMessageBox.information(self, 'Success', 'Image saved successfully!')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitViewerApp()
    window.show()
    sys.exit(app.exec_())