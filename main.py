import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox,
                            QTableWidget, QTableWidgetItem, QDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

# Define the quantization median values for each group (32 groups with a width of 8)
median_values = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124,
                 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]

# Hard-coded mapping for reversion
REVERSION_MAPPING = np.array(median_values, dtype=np.uint8)

class ValueDisplayDialog(QDialog):
    def __init__(self, quantized_values, reduced_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quantization and Bit Reduction Values")
        self.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Original Range", "Quantized Value", "Reduced Value"])
        self.table.setRowCount(len(quantized_values))
        
        for i in range(len(quantized_values)):
            lower_bound = i * 8
            upper_bound = lower_bound + 7
            range_item = QTableWidgetItem(f"{lower_bound}-{upper_bound}")
            quant_item = QTableWidgetItem(str(quantized_values[i]))
            reduced_item = QTableWidgetItem(str(reduced_values[i]))
            
            self.table.setItem(i, 0, range_item)
            self.table.setItem(i, 1, quant_item)
            self.table.setItem(i, 2, reduced_item)
        
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

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
    return np.right_shift(img_array, 3)

def revert_bit_reduction(reduced_bit_image):
    """Revert using direct lookup table mapping"""
    return REVERSION_MAPPING[reduced_bit_image]

def save_as_bin(data, filename):
    """Save numpy array as raw binary file"""
    with open(filename, 'wb') as f:
        np.save(f, data)

def load_from_bin(filename):
    """Load numpy array from binary file"""
    with open(filename, 'rb') as f:
        return np.load(f)

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.current_image_array = None

    def initUI(self):
        self.setWindowTitle('Image Processing App')
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; } 
            QLabel { font-size: 18px; font-weight: bold; color: #ffffff; } 
            QPushButton { 
                background-color: #4CAF50; color: white; border-radius: 10px; 
                padding: 12px; font-size: 16px; border: none; 
            } 
            QPushButton:hover { background-color: #45a049; } 
            QRadioButton { font-size: 16px; color: #ffffff; } 
            #imageLabel { 
                background-color: #333; border: 2px dashed #666; 
                min-width: 250px; min-height: 250px; 
            } 
            QDialog { background-color: #1e1e1e; }
            QTableWidget {
                background-color: #2d2d2d; color: #ffffff; gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px; border: 1px solid #444;
            }
        """)
        
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(50, 50, 50, 50)
        
        self.label_instruction = QLabel('1. Select an image', self)
        self.label_instruction.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_instruction)
        
        self.btn_choose = QPushButton('Choose Image', self)
        self.btn_choose.clicked.connect(self.choose_image)
        self.layout.addWidget(self.btn_choose, alignment=Qt.AlignCenter)
        
        self.label_image = QLabel(self)
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_image, alignment=Qt.AlignCenter)
        
        self.label_options = QLabel('2. Choose an option', self)
        self.label_options.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_options)
        
        self.option1 = QRadioButton('Apply Median Quantization', self)
        self.option2 = QRadioButton('Apply Quantization + Bit Reduction', self)
        self.option3 = QRadioButton('Revert to Quantized Image', self)
        self.layout.addWidget(self.option1, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.option2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.option3, alignment=Qt.AlignCenter)
        
        self.btn_process = QPushButton('Process and Save Image', self)
        self.btn_process.clicked.connect(self.process_and_save_image)
        self.layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)

    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg);;All Files (*)', options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            try:
                img = Image.open(self.image_path)
                self.current_image_array = np.array(img)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load image: {str(e)}')

    def show_values_dialog(self, quantized_values, reduced_values):
        dialog = ValueDisplayDialog(quantized_values, reduced_values, self)
        dialog.exec_()

    def process_and_save_image(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image first.')
            return
        
        option = 1 if self.option1.isChecked() else 2 if self.option2.isChecked() else 3 if self.option3.isChecked() else None
        if option is None:
            QMessageBox.warning(self, 'Error', 'Please choose an option.')
            return
        
        try:
            if self.current_image_array is None:
                img = Image.open(self.image_path)
                img_array = np.array(img)
            else:
                img_array = self.current_image_array
            
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            if option == 1:
                # Median quantization only - save as normal image
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                processed_img_array = np.stack([r_quant, g_quant, b_quant], axis=-1)
                
                save_as, _ = QFileDialog.getSaveFileName(
                    self, 'Save Quantized Image', '', 
                    'PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All Files (*)')
                if save_as:
                    Image.fromarray(processed_img_array.astype(np.uint8)).save(save_as)
                    QMessageBox.information(self, 'Success', f'Quantized image saved to {save_as}')
            
            elif option == 2:
                # Quantization + bit reduction - save ONLY as .bin
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                r_reduced = apply_bit_reduction(r_quant)
                g_reduced = apply_bit_reduction(g_quant)
                b_reduced = apply_bit_reduction(b_quant)
                processed_img_array = np.stack([r_reduced, g_reduced, b_reduced], axis=-1)
                
                # Show quantization values
                reduced_values = [apply_bit_reduction(np.array([val]))[0] for val in median_values]
                self.show_values_dialog(median_values, reduced_values)
                
                # Force .bin extension
                save_as, _ = QFileDialog.getSaveFileName(
                    self, 'Save Bit-Reduced Data', '', 
                    'Binary files (*.bin)')
                
                if save_as:
                    if not save_as.lower().endswith('.bin'):
                        save_as += '.bin'
                    save_as_bin(processed_img_array, save_as)
                    QMessageBox.information(self, 'Success', f'Bit-reduced data saved to {save_as}')
            
            elif option == 3:
                # Revert to quantized image - only allow .bin files
                file_path, _ = QFileDialog.getOpenFileName(
                    self, 'Open Bit-Reduced Data', '', 
                    'Binary files (*.bin)')
                
                if file_path:
                    try:
                        loaded = load_from_bin(file_path)
                        
                        # Verify dimensions
                        if loaded.ndim != 3 or loaded.shape[2] != 3:
                            raise ValueError("Invalid file format - expected 3-channel image data")
                        
                        r_reverted = revert_bit_reduction(loaded[:,:,0])
                        g_reverted = revert_bit_reduction(loaded[:,:,1])
                        b_reverted = revert_bit_reduction(loaded[:,:,2])
                        processed_img_array = np.stack([r_reverted, g_reverted, b_reverted], axis=-1)
                        
                        # Save the reverted image
                        save_as, _ = QFileDialog.getSaveFileName(
                            self, 'Save Reverted Image', '', 
                            'PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All Files (*)')
                        if save_as:
                            Image.fromarray(processed_img_array.astype(np.uint8)).save(save_as)
                            QMessageBox.information(self, 'Success', f'Reverted image saved to {save_as}')
                    
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f"Failed to process binary file: {str(e)}")
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())