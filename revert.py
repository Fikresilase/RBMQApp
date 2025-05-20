import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QFileDialog, QLabel, QVBoxLayout, QWidget, 
                            QTextEdit, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

class BitImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        
    def initUI(self):
        self.setWindowTitle('Bit Image Processor')
        self.setGeometry(100, 100, 900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        self.image_label.setMinimumSize(400, 300)
        
        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: monospace;")
        
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
        
        # Add widgets to main layout
        layout.addWidget(self.image_label)
        layout.addWidget(self.console)
        layout.addLayout(btn_layout)
        
        central_widget.setLayout(layout)
        
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
                
                # Convert to QImage for display
                if pil_img.mode == 'RGB':
                    qimage = QImage(self.original_image.data, 
                                   self.original_image.shape[1], 
                                   self.original_image.shape[0], 
                                   QImage.Format_RGB888)
                elif pil_img.mode == 'RGBA':
                    qimage = QImage(self.original_image.data, 
                                   self.original_image.shape[1], 
                                   self.original_image.shape[0], 
                                   QImage.Format_RGBA8888)
                else:  # Grayscale
                    qimage = QImage(self.original_image.data, 
                                   self.original_image.shape[1], 
                                   self.original_image.shape[0], 
                                   QImage.Format_Grayscale8)
                
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation))
                
                self.btn_process.setEnabled(True)
                self.console.clear()
                self.console.append(f"Loaded image: {file_path}")
                self.console.append(f"Dimensions: {self.original_image.shape}")
                
            except Exception as e:
                self.console.append(f"Error loading image: {str(e)}")
    
    def process_bits(self):
        if self.original_image is None:
            return
            
        try:
            self.console.append("\nProcessing bits...")
            
            # Display original bits (first 5x5 pixels as example)
            sample = self.original_image[:5, :5]
            self.console.append("\nOriginal pixel values (5x5 sample):")
            self.console.append(str(sample))
            
            # Process bits: shift left 3 and add 4
            self.processed_image = (self.original_image << 3) + 4
            
            # Clip values to 0-255 range
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            
            # Display processed bits
            processed_sample = self.processed_image[:5, :5]
            self.console.append("\nProcessed pixel values (5x5 sample):")
            self.console.append(str(processed_sample))
            
            # Display the processed image
            if len(self.processed_image.shape) == 2:  # Grayscale
                qimage = QImage(self.processed_image.data, 
                               self.processed_image.shape[1], 
                               self.processed_image.shape[0], 
                               QImage.Format_Grayscale8)
            else:  # Color
                qimage = QImage(self.processed_image.data, 
                               self.processed_image.shape[1], 
                               self.processed_image.shape[0], 
                               QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation))
            
            self.btn_save.setEnabled(True)
            self.console.append("\nProcessing complete!")
            
        except Exception as e:
            self.console.append(f"Error processing image: {str(e)}")
    
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
                    pil_img = Image.fromarray(self.processed_image, mode='L')
                else:  # Color
                    pil_img = Image.fromarray(self.processed_image, mode='RGB')
                
                # Save with appropriate format based on extension
                if file_path.lower().endswith('.png'):
                    pil_img.save(file_path, 'PNG')
                elif file_path.lower().endswith(('.jpg', '.jpeg')):
                    pil_img.save(file_path, 'JPEG', quality=95)
                else:
                    pil_img.save(file_path)  # Default to PNG
                
                self.console.append(f"\nImage saved to: {file_path}")
                
            except Exception as e:
                self.console.append(f"Error saving image: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitImageProcessor()
    window.show()
    sys.exit(app.exec_())